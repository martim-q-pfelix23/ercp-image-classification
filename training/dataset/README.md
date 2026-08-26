# Local Dataset Directory

The image files used for training are intentionally not committed to Git.

Download MIQR-CC from:

https://doi.org/10.6084/m9.figshare.31079236

Then run `../patient_level_split.ipynb` to create the local structure:

```text
dataset/
├── train/<class>/
├── val/<class>/
└── test/<class>/
```

The archived experiment used 1,568 labelled images after filtering and patient-level splitting.
