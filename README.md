# fastMRI Reconstruction with U-Net

This project implements a baseline U-Net model for MRI reconstruction on the [fastMRI](https://github.com/facebookresearch/fastMRI) single-coil knee dataset. It serves as a starting point for analyzing how targeted loss weighting and attention in U-Nets affects reconstruction image quality. 

Dataset Link: https://fastmri.med.nyu.edu/

## Objective

To explore whether emphasizing the central pixels in the reconstructed image improves the reconstruction quality and SSIM (Structural Similarity Index Measure) by modifying the loss function during training: 
- Build a baseline U-Net model (Model 1)
- Modify the baseline model's loss function to upweight L1 loss in central image regions during training (Model 2)
- Build an Attention U-Net model (Model 3)
- Compare reconstruction quality (SSIM and MSE) of these 3 models

## Folder Structure
```
├── README.md
├── requirements.txt
├── train/
|   ├── fastMRI_knee_SC_baseline.ipynb: Training of baseline
|   ├── fastMRI_knee_SC_center_weighted.ipynb: Training of model with weighted loss
|   ├── unet_knee_sc.py: Training params for baseline model
|   ├── unet_knee_sc_center_weighted_loss.py: Training params for weighted loss model
├── fastMRI/
│   ├── data/ (Data processing for fastMRI dataset)
|   |── models/ (Model definitions)
|   ├── pl_modules/ (Pytorch Lightning data and models modules)
|   |── losses.py: Custom loss functions
|   |── ... 
```
