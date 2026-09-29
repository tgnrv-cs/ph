"""PanEcho teacher adapter (licence: CC BY-NC-SA 4.0, CarDS-Yale/PanEcho) — the optional second teacher.

Weights dataset layout expected under `weights_dir`:
    <weights_dir>/PanEcho/            the PanEcho repository (hubconf.py with the `PanEcho` entry point)
    <weights_dir>/**/panecho*.pt*     the released checkpoint (only needed when hubconf cannot download it; no internet in background runs)

Model: ConvNeXt-T frame encoder + temporal transformer; input 1×3×16×224×224 (ImageNet normalisation).
build() returns (module, 224, dim, preprocess, licence); module(x) -> pooled pre-head clip embedding [B, dim], captured by a forward
hook on the first task head so the adapter is independent of the head naming. Runs in fp16 under torch.no_grad() on a T4.
"""
import glob
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

LICENCE = "CC BY-NC-SA 4.0 (PanEcho, CarDS-Yale/PanEcho)"
INPUT_RES, FRAMES = 224, 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


class PanEchoEmbedder(nn.Module):
    """Runs the full PanEcho model and returns the embedding that enters the task heads (mean over any remaining token axis)."""
    def __init__(self, model: nn.Module):
        super().__init__(); self.model = model; self._emb = None
        heads = next((m for n, m in model.named_modules() if n.split(".")[-1] in ("heads", "task_heads", "head", "fc", "classifier") and n), None)
        if heads is None:
            raise RuntimeError("PanEcho: no task-head module found to hook (expected 'heads' / 'task_heads' / 'head' / 'fc')")
        first = next(iter(heads.children()), heads) if isinstance(heads, (nn.ModuleDict, nn.ModuleList, nn.Sequential)) else heads
        first.register_forward_pre_hook(lambda m, inp: setattr(self, "_emb", inp[0]))
    def forward(self, x):
        self._emb = None; self.model(x); e = self._emb
        assert e is not None, "PanEcho head hook did not fire"
        return e.mean(1) if e.dim() == 3 else e


def build(weights_dir):
    wd = Path(weights_dir); repo = next((p.parent for p in wd.rglob("hubconf.py") if "panecho" in str(p).lower()), None)
    if repo is None: raise FileNotFoundError(f"PanEcho repository with hubconf.py not found under {wd}")
    ck = next((Path(p) for p in glob.glob(str(wd / "**" / "*panecho*.pt*"), recursive=True)), None)
    try:
        model = torch.hub.load(str(repo), "PanEcho", source="local", pretrained=(ck is None))
    except TypeError:
        model = torch.hub.load(str(repo), "PanEcho", source="local")
    if ck is not None:
        sd = torch.load(ck, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd.get("model", sd))
        res = model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
        print(f"[panecho] loaded {ck.name}: {len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected keys")
        assert len(res.unexpected_keys) == 0, "PanEcho checkpoint keys do not match the hub model"
    emb = PanEchoEmbedder(model).eval()
    for p in emb.parameters(): p.requires_grad_(False)
    if torch.cuda.is_available(): emb = emb.half().cuda()
    with torch.no_grad():
        p = next(emb.parameters()); dim = int(emb(torch.zeros(1, 3, FRAMES, INPUT_RES, INPUT_RES, device=p.device, dtype=p.dtype)).shape[1])
    def preprocess(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != INPUT_RES or x.shape[-2] != INPUT_RES:
            B, C, T = x.shape[:3]; x = F.interpolate(x.transpose(1, 2).reshape(B * T, C, *x.shape[-2:]), size=(INPUT_RES, INPUT_RES), mode="bilinear", align_corners=False).view(B, T, C, INPUT_RES, INPUT_RES).transpose(1, 2)
        if x.shape[2] != FRAMES:
            idx = torch.linspace(0, x.shape[2] - 1, FRAMES).round().long().to(x.device); x = x[:, :, idx]
        x = (x - MEAN.to(x.device)) / STD.to(x.device); p = next(emb.parameters()); return x.to(device=p.device, dtype=p.dtype)
    return emb, INPUT_RES, dim, preprocess, LICENCE
