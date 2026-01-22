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
def build_pyramid(self, fmap1, fmap2):
    """Build the correlation pyramid from two feature maps.

        The correlation volume is first computed as the dot product of each pair (pixel_in_fmap1, pixel_in_fmap2)
        The last 2 dimensions of the correlation volume are then pooled num_levels times at different resolutions
        to build the correlation pyramid.
        """
    if fmap1.shape != fmap2.shape:
        raise ValueError(f'Input feature maps should have the same shape, instead got {fmap1.shape} (fmap1.shape) != {fmap2.shape} (fmap2.shape)')
    min_fmap_size = 2 * 2 ** (self.num_levels - 1)
    if any((fmap_size < min_fmap_size for fmap_size in fmap1.shape[-2:])):
        raise ValueError(f'Feature maps are too small to be down-sampled by the correlation pyramid. H and W of feature maps should be at least {min_fmap_size}; got: {fmap1.shape[-2:]}. Remember that input images to the model are downsampled by 8, so that means their dimensions should be at least 8 * {min_fmap_size} = {8 * min_fmap_size}.')
    corr_volume = self._compute_corr_volume(fmap1, fmap2)
    batch_size, h, w, num_channels, _, _ = corr_volume.shape
    corr_volume = corr_volume.reshape(batch_size * h * w, num_channels, h, w)
    self.corr_pyramid = [corr_volume]
    for _ in range(self.num_levels - 1):
        corr_volume = F.avg_pool2d(corr_volume, kernel_size=2, stride=2)
        self.corr_pyramid.append(corr_volume)