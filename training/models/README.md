# Local Model Artifacts

Training notebooks save checkpoints and generated figures to this directory.

Large model files are intentionally excluded from Git to keep the repository lightweight.

Typical local artifacts include:
- `*.pth` / `*.pt` checkpoints;
- training-history plots;
- confusion matrices.

Summary metrics that are suitable for version control are stored under `../../results/`.
