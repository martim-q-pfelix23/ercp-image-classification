# Model Checkpoints

The source archive supplied for this portfolio contains **Git LFS pointer files**, not the binary `.pth` weights themselves.

`checkpoint_manifest.csv` records the original checkpoint paths, SHA-256 LFS object IDs and expected binary sizes for 13 checkpoints (approximately 978.3 MB in total).

## Recommended storage

Do not commit ~1 GB of model binaries directly to normal Git history.

Use one of:

1. **Git LFS** - suitable if the original LFS objects are still available;
2. **GitHub Releases** - good for publishing the final six ensemble checkpoints as a versioned artifact;
3. an external model registry such as Hugging Face.

The final ensemble is reconstructed from the six component checkpoints and `../results/final_ensemble_config.json`.
