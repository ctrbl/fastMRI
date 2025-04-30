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
├── data/ (Download link: https://fastmri.med.nyu.edu/)
|   ├── singlecoil_train/
|   ├── singlecoil_val/
|   ├── singlecoil_test/
├── train/
|   ├── fastMRI_knee_SC_baseline.ipynb: Training of baseline model
|   ├── fastMRI_knee_SC_center_weighted.ipynb: Training of center-weighted loss model
|   ├── fastMRI_knee_SC_AttUnet_center_weighted.ipynb: Training of Attention U-Net
|   ├── unet_knee_sc.py: Training params for baseline model
|   ├── unet_knee_sc_center_weighted_loss.py: Training params for center-weighted loss model
|   ├── unet_knee_sc_AttUnet.py: Training params for Attention U-Net
├── evaluate/
|   ├── evaluate_models.ipynb: Evaluation script to generate reconstruction images
|   ├── test_models.ipynb: Evaluate outputs against target and visualize reconstruction and error images
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
- To train each model, select the corresponding Jupyter notebook file (fix data and log paths if needed)

## Evaluating Models
- Go to ~/evaluate/
- To generate output reconstruction images, run evaluate_models.ipynb (fix data and log paths if needed)
- To visualize target, output reconstructions, and error images, run test_models.ipynb

- To simply get evaluation metrics of reconstruction images, go to ~/fastmri/ and run:
  ```
  python evaluate.py \
  --target-path <TARGET_PATH>
  --predictions-path <RECONSTRUCTION_PATH>
  --challenge singlecoil
  ```
