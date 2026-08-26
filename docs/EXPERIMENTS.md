# Experiment Catalogue

The project evolved from simple transfer-learning baselines to data augmentation, specialist models and an optimized ensemble.

## 01 - Data

- `patient_level_split.ipynb` - patient-aware 70/15/15 split with overlap checks and stratification fallback.

## 02 - Baselines

- ResNet50
- DenseNet121
- MobileNetV2
- EfficientNet-B7
- DeiT-III

## 03 - Architecture Exploration

- ResNet18 / ResNet34 / ResNet50
- DenseNet121
- MobileNetV2
- EfficientNet-B0 / B3 / B5 / V2-S
- ConvNeXt-Small
- DeiT-III
- Swin-Tiny

## 04 - Augmentation

- CLAHE experiment
- perceptual-loss VAEs for Biliary Leaks, Normal and Stricture
- ConvNeXt-Tiny trained with VAE-generated samples
- SwinV2-Tiny trained with VAE-generated samples

The archived VAE generation produced 300 synthetic samples for each of Biliary Leaks, Normal and Stricture.

## 05 - Feature Experiments

A set of XGBoost experiments explores learned/latent representations and alternative losses.

## 06 - Specialist Models

- binary Biliary Leaks specialist
- improved Biliary Leaks specialist
- Stricture-vs-Lithiasis specialist

These experiments investigate whether difficult class-specific errors can be corrected after the main multiclass prediction.

## 07 - Final Ensemble

`final_weighted_ensemble.ipynb` performs validation-driven optimization of a direct weighted ensemble containing:

- EfficientNet-B7
- DenseNet121
- MobileNetV2
- ResNet50
- ConvNeXt-Tiny + VAE
- EfficientNet-B0 + VAE

Final test performance:

- Accuracy: **0.8165**
- Macro F1: **0.7881**

The same notebook also contains final class-wise evaluation, confusion matrix generation, Grad-CAM and occlusion-sensitivity analysis.
