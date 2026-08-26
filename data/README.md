# Data

The original MIQR-CC image files are intentionally not versioned in this portfolio repository.

## Official dataset

MIQR-CC is available from Figshare:

- Dataset DOI: https://doi.org/10.6084/m9.figshare.31079236
- Publication: https://doi.org/10.1038/s41597-026-07679-1
- Original dataset/code repository: https://github.com/monicaccmartins/MIQR-CC-Dataset

The dataset is distributed under CC BY 4.0.

## Exact experimental split

`split_manifest.csv` preserves the exact image filenames, patient identifiers, classes and train/validation/test assignments used by the archived experiments.

The split contains 1,568 labelled images from 436 patients:

| Split | Patients | Images |
|---|---:|---:|
| Train | 305 | 1,067 |
| Validation | 65 | 234 |
| Test | 66 | 267 |

The split is patient-aware: a patient is assigned to exactly one partition.

To reconstruct the folders locally, download MIQR-CC and use the manifest or rerun `notebooks/01_data/patient_level_split.ipynb`.
