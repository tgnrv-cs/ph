# Kaggle experiment notebooks

Two self-contained notebooks implement the two studies of *Kaggle Experiment Designs* (Sep 29, 2026):

| Notebook | Experiment | Stages (one Kaggle session each) | Output datasets |
| --- | --- | --- | --- |
| `K1_quality_gated_anatomy_routing.ipynb` | K1 — quality-gated anatomy routing on CardiacNet-PAH | 1 audit/cache · 2 teacher latents · 3 distillation · 4 cross-fitted segmentation + quality estimator + feature cache · 5 diagnostic arms G0–G4 on cached features · 6 G5 joint + CAGrad, A7/A8, C1/C2 · 7 statistics | `cardiacnet-cache-v1`, `teacher-latents-v1`, `student-init-v1`, `seg-cores-v1`, `diag-features-v1`, `diag-results-v1`, `joint-results-v1`, `ablation-results-v1`, `k1-final-v1` |
| `K2_partially_labelled_federation.ipynb` | K2 — partially labelled four-chamber federation over cached frozen features | 10 harmonise/cache · 11 baselines · 12 method arms + U-Net reference · 13 DP-SGD, secure aggregation, membership inference, conformal margins · 14 student, statistics | `fed-features-v1`, `fed-baselines-v1`, `fed-method-v1`, `fed-privacy-v1`, `k2-final-v1` |

## Running a stage on Kaggle

1. Attach the raw data (CardiacNet from its Kaggle listing; CAMUS and HMC-QU as private datasets; teacher / encoder weights as
   private datasets) and the output dataset(s) of the previous stage(s). The first cell of every stage prints what it depends on and
   refuses to run if a `VERSION.json` is missing.
2. Select the work for this session with environment variables (or edit `CFG` in cell 1):

   ```bash
   # K1 — session S4: segmentation cross-fit for folds 0–2 with the distilled init
   K1_STAGES=4 K1_FOLDS=0,1,2 K1_INITS=D1
   # K1 — session S6: diagnostic arms, all folds, three seeds, both inits
   K1_STAGES=5 K1_FOLDS=0,1,2,3,4 K1_SEEDS=0,1,2 K1_INITS=D1,D0
   # K2 — session S13: privacy stage, three seeds
   K2_STAGES=13 K2_SEEDS=0,1,2
   ```

   Data locations default to `/kaggle/input/{cardiacnet,camus,hmc-qu,echojepa-weights,dinov3-weights,panecho-weights}` and can be
   overridden with `CARDIACNET_ROOT`, `CAMUS_ROOT`, `HMCQU_ROOT`, `TEACHER_WEIGHTS`, `DINOV3_DIR`, `PANECHO_DIR`.
3. "Save version → Save and run all". Every long run checkpoints each epoch and resumes; a `runs_manifest.json` per stage lists
   completed fold–seed–arm combinations, so a restarted session picks up the first missing one. Each run writes one CSV under
   `<dataset>/runs/` with metrics, timing, peak memory, the Kaggle image tag and a hash of the notebook source.
4. Save `/kaggle/working` as the stage's dataset version and attach it to the next stage's notebook.

## What still needs your confirmation before real sessions

- **Dataset layouts and label ids.** `iter_cardiacnet` (both notebooks), `iter_camus`, `iter_hmcqu` and the extras loader follow the
  published releases as far as they are known; verify `CFG.cardiacnet_label_map`, `CFG.cardiacnet_time_axis` and the file layout
  against the actual Kaggle listing on the first S1/S10 session.
- **Teacher weights (K1 stage 2).** EchoJEPA-L / PanEcho are loaded through an adapter file `load.py` in the weights dataset that
  returns `(module, input_res, dim, preprocess)`; see `HubTeacher`.
- **Frozen encoders (K2 stage 10).** `DINOv3Encoder` expects the dinov3 repo (torch.hub `source='local'`) plus a `dinov3_vits16*.pth`
  checkpoint; `PanEchoEncoder` loads a ConvNeXt-T state dict through `timm`.
- **C1 CaRDNet-X Lite (K1 stage 6)** is a size-matched stand-in; paste your original definition into `CaRDNetXLite`.
- **D0 init** is "no distillation" (Kaiming). An ImageNet/Kinetics stem cannot be mapped onto the depthwise-separable R(2+1)D blocks
  automatically; a compatible state dict can be supplied with `D0_WEIGHTS`.

## Validation performed

Both notebooks were executed end to end with `nbclient` on a CPU-only machine in **smoke mode** (`SMOKE_TEST=1`, automatic when
`/kaggle/input` is absent). Smoke mode synthesises small CardiacNet / CAMUS / HMC-QU-like corpora with the correct label sets,
acquisition groups, duplicates and a contradictory label, and swaps the teacher / frozen encoders for seeded random networks.
Every stage, arm, statistic and figure runs (K1 ≈ 7 min, K2 ≈ 7 min at 64 px); the committed notebooks carry those outputs.
Smoke numbers validate the code paths only and carry no scientific meaning.

Checks encoded in the notebooks themselves: the parameter budget (`SegModel` ≤ 5 M, measured 0.86 M; decoder + adapters 0.76 M ≤
1.2 M; U-Net reference 7.8 M; student 1.9 M ≤ 10 M), the manifest hash on every load, the secure-aggregation equality check on every
DP round, and the withdrawal conditions of the design doc, which are evaluated automatically and printed as findings.

`pip install -r ../requirements.txt` reproduces the local environment.
