"""EchoJEPA-L teacher adapter (licence: MIT, bowang-lab/EchoJEPA).

Weights dataset layout expected under `weights_dir`:
    <weights_dir>/EchoJEPA/           the EchoJEPA (V-JEPA 2-style) repository, with src/models/vision_transformer.py
    <weights_dir>/**/echojepa*.pt*    a checkpoint holding the target encoder (keys 'target_encoder' or 'encoder')

Model: ViT-L/16, 16 frames × 224², tubelet 2×16×16 → 8 × 14 × 14 = 1568 tokens of dim 1024.
build() returns (module, 224, 1024, preprocess, licence). module(x) -> pooled clip embedding [B, 1024] (mean over target-encoder
patch tokens); module.tokens(x) -> token grid [B, T/2, 14, 14, 1024] for token-level distillation. Runs in fp16 under
torch.no_grad() on a T4 (the caller wraps the forward in no_grad / autocast).
"""
import sys, glob
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

LICENCE = "MIT (EchoJEPA-L, bowang-lab/EchoJEPA)"
INPUT_RES, FRAMES, DIM, TUBELET, PATCH = 224, 16, 1024, 2, 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def _find(weights_dir: Path):
    repo = next((p.parent.parent.parent for p in Path(weights_dir).rglob("src/models/vision_transformer.py")), None)
    ck = next((Path(p) for pat in ("**/echojepa*.pt*", "**/*jepa*.pt*", "**/*.pth.tar") for p in glob.glob(str(Path(weights_dir) / pat), recursive=True)), None)
    if repo is None or ck is None:
        raise FileNotFoundError(f"EchoJEPA repo (src/models/vision_transformer.py) or checkpoint not found under {weights_dir}")
    return repo, ck


def _strip(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        for pre in ("module.", "backbone.", "encoder.", "model."):
            if k.startswith(pre): k = k[len(pre):]
        out[k] = v
    return out


class EchoJEPAWrapper(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__(); self.encoder = encoder
    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        t = self.encoder(x)                                          # [B, N, D]
        B, N, D = t.shape; T = x.shape[2] // TUBELET; hw = int(round((N // T) ** 0.5))
        return t.view(B, T, hw, hw, D)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).mean(1)


def build(weights_dir):
    repo, ck = _find(Path(weights_dir)); sys.path.insert(0, str(repo))
    try:                                                             # preferred: the repo's own hub entry point
        enc = torch.hub.load(str(repo), "echojepa_vitl", source="local", pretrained=False)
    except Exception:
        from src.models.vision_transformer import vit_large        # V-JEPA 2 code path
        enc = vit_large(img_size=INPUT_RES, patch_size=PATCH, num_frames=FRAMES, tubelet_size=TUBELET, uniform_power=True, use_sdpa=True, use_rope=True)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("target_encoder", sd.get("encoder", sd.get("state_dict", sd))); sd = _strip(sd)
    res = enc.load_state_dict(sd, strict=False)
    print(f"[echojepa_l] loaded {ck.name}: {len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected keys")
    assert len(res.unexpected_keys) == 0 and len(res.missing_keys) <= 2, "checkpoint does not match the ViT-L/16 target encoder; inspect the keys"
    model = EchoJEPAWrapper(enc).eval()
    for p in model.parameters(): p.requires_grad_(False)
    if torch.cuda.is_available(): model = model.half().cuda()
    def preprocess(x: torch.Tensor) -> torch.Tensor:                 # x [B,3,T,H,W] in [0,1] (grayscale already repeated to 3 channels)
        if x.shape[-1] != INPUT_RES or x.shape[-2] != INPUT_RES:
            B, C, T = x.shape[:3]; x = F.interpolate(x.transpose(1, 2).reshape(B * T, C, *x.shape[-2:]), size=(INPUT_RES, INPUT_RES), mode="bilinear", align_corners=False).view(B, T, C, INPUT_RES, INPUT_RES).transpose(1, 2)
        if x.shape[2] != FRAMES:
            idx = torch.linspace(0, x.shape[2] - 1, FRAMES).round().long().to(x.device); x = x[:, :, idx]
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        p = next(model.parameters()); return x.to(device=p.device, dtype=p.dtype)
    return model, INPUT_RES, DIM, preprocess, LICENCE
