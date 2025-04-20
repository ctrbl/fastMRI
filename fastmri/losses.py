"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        reduced: bool = True,
    ):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if reduced:
            return 1 - S.mean()
        else:
            return 1 - S


class CenterWeightedL1Loss(nn.Module):
    
    def __init__(self,center_frac = 0.25,center_weight =3.0):
        """
        Args:
            center_frac: Fraction (0 to 1) of the image height/width to treat as the "center" region.
            center_weight: Weight multiplier for the center region.
        """
        
        super().__init__()
        self.center_frac = center_frac
        self.center_weight = center_weight
        
    
    def forward(self,input,target):
        assert input.shape == target.shape, "Shape mismatch between input and target"
        #print(input.shape)
        b, h, w = input.shape

        # Create weighting mask
        weight_mask = torch.ones_like(input)

        h_start = int(h * (0.5 - self.center_frac / 2))
        h_end = int(h * (0.5 + self.center_frac / 2))
        w_start = int(w * (0.5 - self.center_frac / 2))
        w_end = int(w * (0.5 + self.center_frac / 2))

        weight_mask[:, h_start:h_end, w_start:w_end] *= self.center_weight

        # Compute weighted L1 loss
        loss = F.l1_loss(input, target, reduction='none')
        weighted_loss = (loss * weight_mask).mean()

        return weighted_loss

        
        
    
    