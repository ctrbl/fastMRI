# fastMRI Reconstruction with U-Net

This project implements a baseline U-Net model for image reconstruction on the [fastMRI](https://github.com/facebookresearch/fastMRI) [single-coil knee dataset](https://fastmri.med.nyu.edu/). It serves as a starting point for analyzing how targeted loss weighting affects reconstruction image quality. 

## Objective

To explore whether emphasizing the central pixels in the reconstructed image improves the reconstruction quality and SSIM (Structural Similarity Index Measure) by modifying the loss function during training: 
- Build a baseline U-Net model
- Modify the loss function to upweight L1 loss in central image regions
- Compare reconstruction quality (SSIM and MSE) against baseline

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
