# Preprocessing

This folder contains the preprocessing notebook used to inspect and segment MIQR-CC ERCP fluoroscopy images.

The notebook includes:
- Canny edge detection;
- Hough-based vertical-line detection;
- segmentation using vertical boundaries;
- heuristic segment validation;
- visual inspection utilities.

The MIQR-CC dataset itself is not stored in this repository. Download it from:

https://doi.org/10.6084/m9.figshare.31079236

Update the metadata/image paths in the notebook or set `METADATA_PATH` before running it.
