import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from .expanded_weights_utils import \
def conv_padding_for_same(dilation, kernel_size):
    total_pad = dilation * (kernel_size - 1)
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    return (left_pad, right_pad)