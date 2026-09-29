# Kaggle experiment notebooks

Two self-contained notebooks implement the two studies of *Kaggle Experiment Designs* (Sep 29, 2026):

| Notebook | Experiment | Stages (one Kaggle session each) | Output datasets |
| --- | --- | --- | --- |
| `K1_quality_gated_anatomy_routing.ipynb` | K1 — quality-gated anatomy routing on CardiacNet-PAH | 0 label/axis audit gate · 1 audit/cache · 2 teacher latents · 3 distillation · 4 cross-fitted quality targets + final core + feature cache · 5 diagnostic arms G0–G4 on cached features · 6 G5 joint + CAGrad, A0/A7/A8, C1/C2 · 7 statistics | `cardiacnet-cache-v1`, `teacher-latents-v1`, `student-init-v1`, `seg-cores-v1`, `diag-features-v1`, `diag-results-v1`, `joint-results-v1`, `ablation-results-v1`, `k1-final-v1` |
| `K2_partially_labelled_federation.ipynb` | K2 — partially labelled four-chamber federation over cached frozen features | 10 harmonise/cache · 11 baselines · 12 method arms + U-Net reference · 13 video-level DP-SGD, secure aggregation, membership inference, conformal margins · 14 federated student, statistics | `fed-features-dinov3-v1`, `fed-features-panecho-v1`, `fed-baselines-v1`, `fed-method-v1`, `fed-privacy-v1`, `k2-final-v1` |

## Confirmation gates (must be passed before the first real session)

| Gate | Evidence produced | How to pass |
| --- | --- | --- |
| Chamber-id map and NIfTI time axis (K1 Stage 0, CPU) | `stage0/label_id_<id>.png` overlay panels per raw label id, centroid/area statistics, a suggested mapping from apex-up A4C anatomy, shapes of five files with the chosen frame axis | set `CFG.cardiacnet_label_map` and `CFG.cardiacnet_time_axis` explicitly, then `K1_LABEL_MAP_CONFIRMED=1`. Stage 1 refuses to run otherwise. |
| Acquisition grouping (K1 Stage 1) | the 20 most frequent filename tokens, the group-size histogram and `n_groups` under `CFG.group_scope` (`folder` = `branch/class_folder/token`, default; `token` = merged across folders and branches) | `250 ≤ n_groups ≤ n_videos` is asserted, then `K1_GROUPING_CONFIRMED=1`. Folds are drawn once over the union of the PAH and ASD branches, stratified on branch × label, so a group never straddles folds. |
| Frozen encoders (K2 Stage 10) | forward of one real frame, printed feature shapes, PanEcho matched-tensor count (must be 100 % or the stage aborts), PCA-3 false-colour image `selftest_<encoder>.png` | inspect the image; nothing to set. |

Both gates are bypassed only in smoke mode.

## Running a stage on Kaggle

1. Attach the raw data (CardiacNet from its Kaggle listing; CAMUS and HMC-QU as private datasets), the weight datasets, the `code`
   datasets, and the output dataset(s) of the previous stage(s). The first cell of every stage prints what it depends on and refuses to
   run if a `VERSION.json` is missing.
2. Select the session's work with environment variables (or edit `CFG` in cell 1):

   | Session | Setting |
   | --- | --- |
   | K1 S0 (CPU) | `K1_STAGES=0` |
   | K1 S1 (CPU) | `K1_STAGES=1 K1_LABEL_MAP_CONFIRMED=1 K1_GROUPING_CONFIRMED=1` |
   | K1 S2 | `K1_STAGES=2 TEACHER_NAME=echojepa_l` (second teacher: `TEACHER_NAME=panecho`, separate dataset version) |
   | K1 S3 | `K1_STAGES=3` |
   | K1 S4 | `K1_STAGES=4 K1_FOLDS=0,1,2 K1_INITS=D1` |
   | K1 S5 | `K1_STAGES=4 K1_FOLDS=3,4 K1_INITS=D1` then `K1_STAGES=4 K1_FOLDS=0,1,2,3,4 K1_INITS=D0` (D0 = final segmenter only, 5 runs) |
   | K1 S6 | `K1_STAGES=5 K1_FOLDS=0,1,2,3,4 K1_SEEDS=0,1,2 K1_INITS=D1,D0` |
   | K1 S7 | `K1_STAGES=6` with the G5 block only (comment out the screening loop or run S7/S8 in one session if the measured G5 time allows) |
   | K1 S8 | `K1_STAGES=6 K1_PROMOTE=1` |
   | K1 S9 | `K1_STAGES=7` |
   | K1 capacity ablation | `K1_CAPACITY=small` re-runs stages 3–5 into separate dataset versions (screening family) |
   | K2 S10 | `K2_STAGES=10 K2_ENCODERS=dinov3,panecho` (one output dataset per encoder) |
   | K2 S11 | `K2_STAGES=11 K2_SEEDS=0,1,2` |
   | K2 S12 | `K2_STAGES=12 K2_SEEDS=0,1,2` |
   | K2 S13 | `K2_STAGES=13 K2_SEEDS=0,1,2` |
   | K2 S14 | `K2_STAGES=14` |

   Data locations default to `/kaggle/input/{cardiacnet,camus,hmc-qu,echojepa-weights,dinov3-weights,panecho-weights,cardnet-x-lite-code}`
   and can be overridden with `CARDIACNET_ROOT`, `CAMUS_ROOT`, `HMCQU_ROOT`, `TEACHER_WEIGHTS`, `DINOV3_DIR`, `PANECHO_DIR`, `CARDNET_CODE_DIR`.
3. "Save version → Save and run all". Every long run checkpoints each epoch and resumes; a `runs_manifest.json` per stage lists
   completed fold–seed–arm combinations, so a restarted session picks up the first missing one. Each run writes one CSV under
   `<dataset>/runs/` with metrics, timing, peak memory, the Kaggle image tag and a hash of the notebook source.
4. Save `/kaggle/working` as the stage's dataset version and attach it to the next stage's notebook.

## Session budget (fill in after the first real S4 fold)

The design doc's 0.4 h/run figure was an extrapolation from a 2.76 M model at 128²; the default core is now 3.4 M parameters with a
0.7 M decoder and the cache holds four 16-frame samplings per video, so per-run times must be **measured**. Stage 4 logs
`sec_per_epoch` for every segmenter run; multiply by `CFG.seg_epochs` and re-cut the sessions if a stage runs more than 30 % over.

| Session | Runs | Measured h/run | Measured GPU-h | Plan (design doc) |
| --- | --- | --- | --- | --- |
| S3 distillation | 1 | — | — | 5 |
| S4 segmentation folds 0–2 | 12 | — | — | 5 |
| S5 segmentation folds 3–4 + D0 finals | 8 + 5 | — | — | 6 |
| S6 diagnostic arms | 75 + 15 (D0 G1) | — | — | 6 |
| S7 G5 joint | 15 | — | — | 9 |
| S8 screening | 25 (+ promotions) | — | — | 9 |
| S11–S13 federation | seconds per round over cached features | — | — | 5 + 7 + 6 |

## Arm families

| Family | Arms | Seeds |
| --- | --- | --- |
| K1 confirmatory | G2 vs G1, G2 vs G3, G2 vs G0, D1 vs D0 (on the G1 entropy-gate arm; with `K1_D0_FULL=1` also on G2) | 5 folds × 3 seeds |
| K1 screening | G5 (joint + CAGrad, fixed per-video q̂ shared with G2), A0/A7/A8 input ablations, C1 (original CaRDNet-X Lite), C2 (Kinetics R3D-18), C4 (stage-4 final segmenter), `small` capacity variant | 5 folds × 1 seed, promoted to 3 seeds when the effect exceeds seed noise |
| K2 | local-only, centralised (bg / marginal), FedAvg, FedProx, SCAFFOLD, FedInit, Fed-Consist, FedPSL-style, FedAvg + marginal, + ConDist, + FedBN, encoder swap, U-Net full-model reference, GroupNorm reference, DP-SGD at ε ∈ {1, 4, 8}, federated student, centralised student (oracle) | 3 seeds × 100 rounds |

SCAFFOLD runs SGD (lr × 20, no momentum) because its control-variate correction is defined for SGD; every other arm uses Adam. The
`optimizer` column of the arm table records this.

## Licences and weights

| Weights | Licence | Used for |
| --- | --- | --- |
| EchoJEPA-L (`teacher_adapters/echojepa_l.py`) | MIT | K1 stage 2 teacher |
| PanEcho (`teacher_adapters/panecho.py`, `PanEchoEncoder`) | CC BY-NC-SA 4.0 | K1 second teacher; K2 echo-specific frozen encoder |
| DINOv3 ViT-S/16 (`DINOv3Encoder`) | research licence | K2 primary frozen encoder |
| Kinetics-400 R3D-18 (`r3d_18-*.pth` in the teacher-weights dataset) | BSD (torchvision) | K1 C2 |
| CaRDNet-X Lite (`cardnet-x-lite-code` dataset, `CARDNET_IMPORT=module:Class`) | user's own | K1 C1 (parameter count asserted within ±2 % of 2.76 M) |

EchoFM (CC BY-NC-ND) is not used. No EchoNet-Dynamic or PhysioNet data appears in either notebook. The adapters are exercised by
`teacher_adapters/test_adapters.py` when `TEACHER_WEIGHTS` is present (`python -m pytest experiments/teacher_adapters -q`).

## Design decisions worth knowing

- **Diagnostic features (K1).** The inner cross-fit exists only to produce out-of-sample masks for the quality estimator. Features for every
  labelled video of a fold, with or without masks, come from that fold's final frozen core; training-video q̂ is therefore mildly
  optimistic (recorded in `summary.json`). G4 (oracle) falls back to q̂ where a video has no masks and reports the true-oracle count.
- **Quality estimator (K1).** Two estimators are fitted on the same OOS targets: gradient boosting on CPU morphology features and a GPU MLP
  head on pooled chamber features. The MLP is used when CPU feature extraction exceeds 100 ms per clip and its ρ is within 0.05 of the
  best; both ρ values are reported.
- **Model selection (K1).** Every trained arm, cached-feature (stage 5) and raw-video (stage 6) alike, early-stops on validation NLL with
  validation PAH AUROC as tie-break on a 20 % group-wise split of the training fold; both values are logged per run.
- **Flips (K1).** `CFG.flip_mode` defaults to `none`; `swap` mirrors the frame and swaps LV↔RV / LA↔RA ids. Stage 4 logs the fraction of
  videos whose predicted LV centroid lies right of the RV centroid.
- **FedInit (K2)** relaxes each client's start point away from its last *local* state; smoke mode asserts FedInit ≠ FedAvg.
- **DP unit (K2)** is the video: one sample = K frames (5 CardiacNet, 4 CAMUS, 4 HMC-QU) with the loss summed over frames, per-video
  clipping, Poisson sampling over videos, δ = 1/N_videos, RDP accountant composed over all rounds. `(ε target, ε spent, δ, unit, C, σ,
  steps, accountant)` are written to `VERSION.json` and `summary.json`.
- **Feature storage (K2)** is one fp16 memmap per scale per encoder (`s8.npy`, `s16.npy`) plus `frames.npy`, `labels.npy` and `index.csv`;
  one dataset version per encoder.
- **CAMUS test set (K2)** is the official 50 patients (451–500 or the `testing/` folder); calibration is carved from the 450.

## Validation performed

Both notebooks were executed end to end with `nbclient` on a CPU-only machine in **smoke mode** (`SMOKE_TEST=1`, automatic when
`/kaggle/input` is absent). Smoke mode synthesises small CardiacNet / CAMUS / HMC-QU-like corpora with the correct label sets,
acquisition groups, duplicates and a contradictory label, and swaps the teacher / frozen encoders for seeded random networks. Every
stage, arm, statistic, gate message and figure runs; the committed notebooks carry those outputs. Smoke numbers validate the code paths
only and carry no scientific meaning.

Checks encoded in the notebooks: the parameter budget (`SegModel` ≤ 5 M, measured 3.41 M for `default`, 0.94 M for `small`; K2 decoder
+ adapters 0.76 M ≤ 1.2 M; U-Net reference 7.8 M; student 1.9 M ≤ 10 M), the manifest hash on every load, `source_pred == "final"` for
every cached feature row, FedInit ≠ FedAvg, the secure-aggregation equality on every DP round, and the withdrawal conditions of the design
doc, which are evaluated automatically and printed as findings.

`pip install -r ../requirements.txt` reproduces the local environment.
