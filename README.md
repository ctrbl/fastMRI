# fastMRI Reconstruction with U-Net

This project implements a baseline U-Net model for MRI reconstruction on the [fastMRI](https://github.com/facebookresearch/fastMRI) single-coil knee dataset. It serves as a starting point for analyzing how targeted loss weighting and attention in U-Nets affects reconstruction image quality. 

Dataset Link: https://fastmri.med.nyu.edu/

## Objective

To explore whether targeted weighting loss function and architectural modifications improve the reconstruction quality measured by MSE, NMSE, PSNR, and SSIM scores by implementing these 3 models:
1. Baseline U-Net model
2. Baseline U-Net with center-weighted L1 loss function during training
3. Attention U-Net with center-weighted L1 loss function during training

Due to hidden ground-truth reconstruction images in the test set for the 2020 fastMRI challenge, we will use the untouched portion of the validation/held-out set as the surrogate test set for model evaluation. Note that we used a fixed seed to sample training and validation sets by volume. 

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
- Go to ~/train/
- To tune hyperparameters for model training, select the corresponding python file
- To train each model, select the corresponding Jupyter notebook file (fix directory paths)

## Evaluating Models
- Go to ~/evaluate/
- Run evaluate_models.ipynb (fix directory paths) to generate output reconstruction images
- To evaluate against target reconstruction images, go to ~/fastmri/ and run:
  ```
  python evaluate.py \
  --target-path <TARGET_PATH>
  --predictions-path <RECONSTRUCTION_PATH>
  --challenge singlecoil
  ```
