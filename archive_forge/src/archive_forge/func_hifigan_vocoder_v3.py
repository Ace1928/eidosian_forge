from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
def hifigan_vocoder_v3() -> HiFiGANVocoder:
    """Builds HiFiGAN Vocoder with V3 architecture :cite:`NEURIPS2020_c5d73680`.

    Returns:
        HiFiGANVocoder: generated model.
    """
    return hifigan_vocoder(upsample_rates=(8, 8, 4), upsample_kernel_sizes=(16, 16, 8), upsample_initial_channel=256, resblock_kernel_sizes=(3, 5, 7), resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)), resblock_type=2, in_channels=80, lrelu_slope=0.1)