# Training Experiments

This folder contains the patient-level split pipeline and the model-training notebooks.

## Recommended order

1. `patient_level_split.ipynb`
2. Choose one or more architecture notebooks:
   - `resnet50.ipynb`
   - `densenet121.ipynb`
   - `efficientnet_b7.ipynb`
   - `mobilenet_v2.ipynb`
   - `deit3_small.ipynb`

The split notebook creates non-overlapping train/validation/test patient sets and exports images to `dataset/`.

Model checkpoints are written to `models/` during training but are intentionally ignored by Git.

See the root `README.md` for methodology, results and reproduction instructions.
