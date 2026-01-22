import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from .expanded_weights_utils import \
def conv_normalizer(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return ((input, weight), {'bias': bias, 'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups})