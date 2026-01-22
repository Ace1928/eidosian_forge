import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def get_conv_pool_args_2d_common(self, kernel_size, strides, paddings, dilations, group_num):
    kernels = list(kernel_size)
    assert len(kernels) == 2
    assert len(strides) == 2
    assert len(paddings) == 2
    assert len(dilations) == 2
    ph, pw = paddings
    real_paddings = [ph, ph, pw, pw]
    return ConvPoolArgs2d(*kernels + strides + real_paddings + dilations + [group_num])