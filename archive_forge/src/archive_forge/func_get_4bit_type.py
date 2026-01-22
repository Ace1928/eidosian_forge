import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
from .cextension import COMPILED_WITH_CUDA, lib
def get_4bit_type(typename, device=None, blocksize=64):
    if device is None:
        device = 'cuda'
    data = None
    if typename == 'nf4':
        ' Implements the NF4 data type.\n\n            Constructs a quantization data type where each bin has equal area under a standard normal distribution N(0, 1) that\n            is normalized into the range [-1, 1].\n\n            For more information read the paper: QLoRA: Efficient Finetuning of Quantized LLMs (https://arxiv.org/abs/2305.14314)\n\n            Implementation of the NF4 data type in bitsandbytes can be found in the `create_normal_map` function in\n            the `functional.py` file: https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L236.\n        '
        data = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
    elif typename == 'fp4':
        data = [0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0]
    elif typename == 'int4':
        data = [7, 6, 5, 4, 3, 2, 1, 0, -0, -1, -2, -3, -4, -5, -6, -7]
    elif typename == 'af4':
        if blocksize == 64:
            data = [-1.0, -0.69441008, -0.51243739, -0.3736951, -0.25607552, -0.14982478, -0.04934812, 0.0, 0.04273164, 0.12934483, 0.21961274, 0.31675666, 0.42563882, 0.55496234, 0.72424863, 1.0][::-1]
        else:
            raise NotImplementedError('4-bit AbnormalFloats currently only support blocksize 64.')
    if data is None:
        raise NotImplementedError(f'Typename {typename} not supported')
    data = Tensor(data)
    data /= data.abs().max()
    assert data.numel() == 16
    return data.to(device)