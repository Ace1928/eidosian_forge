from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torchvision.ops import Conv2dNormActivation
from ...transforms._presets import OpticalFlow
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._utils import handle_legacy_interface
from ._utils import grid_sample, make_coords_grid, upsample_flow
def index_pyramid(self, centroids_coords):
    """Return correlation features by indexing from the pyramid."""
    neighborhood_side_len = 2 * self.radius + 1
    di = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
    dj = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
    delta = torch.stack(torch.meshgrid(di, dj, indexing='ij'), dim=-1).to(centroids_coords.device)
    delta = delta.view(1, neighborhood_side_len, neighborhood_side_len, 2)
    batch_size, _, h, w = centroids_coords.shape
    centroids_coords = centroids_coords.permute(0, 2, 3, 1).reshape(batch_size * h * w, 1, 1, 2)
    indexed_pyramid = []
    for corr_volume in self.corr_pyramid:
        sampling_coords = centroids_coords + delta
        indexed_corr_volume = grid_sample(corr_volume, sampling_coords, align_corners=True, mode='bilinear').view(batch_size, h, w, -1)
        indexed_pyramid.append(indexed_corr_volume)
        centroids_coords = centroids_coords / 2
    corr_features = torch.cat(indexed_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous()
    expected_output_shape = (batch_size, self.out_channels, h, w)
    if corr_features.shape != expected_output_shape:
        raise ValueError(f'Output shape of index pyramid is incorrect. Should be {expected_output_shape}, got {corr_features.shape}')
    return corr_features