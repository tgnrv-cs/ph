"""Builds each teacher adapter and forwards a random 16-frame clip. Skipped unless TEACHER_WEIGHTS points at the weights dataset.
Run: TEACHER_WEIGHTS=/kaggle/input/echojepa-weights python -m pytest experiments/teacher_adapters/test_adapters.py -q"""
import os, importlib
from pathlib import Path
import pytest, torch

WEIGHTS = os.environ.get("TEACHER_WEIGHTS", "/kaggle/input/echojepa-weights")
pytestmark = pytest.mark.skipif(not Path(WEIGHTS).exists(), reason="teacher weights dataset not attached")


@pytest.mark.parametrize("name,expect_dim", [("echojepa_l", 1024), ("panecho", None)])
def test_adapter_forward(name, expect_dim):
    mod = importlib.import_module(f"experiments.teacher_adapters.{name}" if Path("experiments").exists() else name)
    model, res, dim, preprocess, licence = mod.build(WEIGHTS)
    assert licence and res == 224 and (expect_dim is None or dim == expect_dim)
    x = torch.rand(2, 3, 16, 112, 112)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        z = model(preprocess(x))
    assert z.shape == (2, dim) and torch.isfinite(z.float()).all()
    if hasattr(model, "tokens"):
        with torch.no_grad():
            t = model.tokens(preprocess(x))
        assert t.shape == (2, 8, 14, 14, dim)
