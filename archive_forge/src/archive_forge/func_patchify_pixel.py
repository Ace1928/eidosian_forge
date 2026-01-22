import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_tvlt import TvltConfig
def patchify_pixel(self, pixel_values):
    """
        pixel_values: [batch_size, num_frames, 3, height, width]
        """
    batch_size, num_frames, num_channels, height, width = pixel_values.shape
    num_patches_height = pixel_values.shape[3] // self.image_patch_size[0]
    num_patches_width = pixel_values.shape[4] // self.image_patch_size[1]
    patchified_pixel_values = pixel_values.reshape(shape=(batch_size, num_frames, num_channels, num_patches_height, self.image_patch_size[0], num_patches_width, self.image_patch_size[1]))
    patchified_pixel_values = torch.einsum('ntchpwq->nthwpqc', patchified_pixel_values)
    patchified_pixel_values = patchified_pixel_values.reshape(shape=(batch_size, num_patches_height * num_patches_width * num_frames, self.image_patch_size[0] * self.image_patch_size[1] * num_channels))
    return patchified_pixel_values