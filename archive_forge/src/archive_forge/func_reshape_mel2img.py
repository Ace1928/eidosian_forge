import collections
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig
def reshape_mel2img(self, normalized_input_features):
    """
        The input is 4 normalized log mel spectrograms. It is reshape to the common shape of images. Each channel
        should represent 1 of the 4 crops of the spectrogram. For more details, refer to the [`ClapFeatureExtractor`].
        """
    _, _, time_length, freq_length = normalized_input_features.shape
    spec_width = int(self.spec_size * self.freq_ratio)
    spec_heigth = self.spec_size // self.freq_ratio
    if time_length > spec_width or freq_length > spec_heigth:
        raise ValueError('the wav size should be less than or equal to the swin input size')
    if time_length < spec_width:
        normalized_input_features = nn.functional.interpolate(normalized_input_features, (spec_width, freq_length), mode='bicubic', align_corners=True)
    if freq_length < spec_heigth:
        normalized_input_features = nn.functional.interpolate(normalized_input_features, (time_length, spec_heigth), mode='bicubic', align_corners=True)
    batch, channels, time, freq = normalized_input_features.shape
    normalized_input_features = normalized_input_features.reshape(batch, channels * self.freq_ratio, time // self.freq_ratio, freq)
    normalized_input_features = normalized_input_features.permute(0, 1, 3, 2).contiguous()
    normalized_input_features = normalized_input_features.reshape(batch, channels, freq * self.freq_ratio, time // self.freq_ratio)
    return normalized_input_features