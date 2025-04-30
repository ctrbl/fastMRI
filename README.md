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
|   ├── fastMRI_knee_SC_AttUnet.ipynb: Training of Attention U-Net
|   ├── unet_knee_sc.py: Training params for baseline model
|   ├── unet_knee_sc_center_weighted_loss.py: Training params for center-weighted loss model
|   ├── unet_knee_sc_AttUnet.py: Training params for Attention U-Net
├── evaluate/
|   ├── evaluate_models.ipynb: 
├── fastMRI/
│   ├── data/ (Data processing for fastMRI dataset)
|   |── models/ (Model definitions)
|   ├── pl_modules/ (Pytorch Lightning data and models modules)
|   |── losses.py: Custom loss functions
|   |── ... 
```

## Training Models
- Go to train folder
- To tune hyperparameters for model training, select the corresponding python file
- To train each model, select the corresponding Jupyter notebook file

## Evaluating Models
- Go to evaluate folder
- Run evaluate_models.ipynb
