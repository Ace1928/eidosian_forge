import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_vit_mae import ViTMAEConfig
def patchify(self, pixel_values):
    """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
    patch_size, num_channels = (self.config.patch_size, self.config.num_channels)
    if pixel_values.shape[2] != pixel_values.shape[3] or pixel_values.shape[2] % patch_size != 0:
        raise ValueError('Make sure the pixel values have a squared size that is divisible by the patch size')
    if pixel_values.shape[1] != num_channels:
        raise ValueError('Make sure the number of channels of the pixel values is equal to the one set in the configuration')
    batch_size = pixel_values.shape[0]
    num_patches_one_direction = pixel_values.shape[2] // patch_size
    patchified_pixel_values = pixel_values.reshape(batch_size, num_channels, num_patches_one_direction, patch_size, num_patches_one_direction, patch_size)
    patchified_pixel_values = torch.einsum('nchpwq->nhwpqc', patchified_pixel_values)
    patchified_pixel_values = patchified_pixel_values.reshape(batch_size, num_patches_one_direction * num_patches_one_direction, patch_size ** 2 * num_channels)
    return patchified_pixel_values