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
def quantize_nf4(A: Tensor, absmax: Optional[torch.Tensor]=None, out: Optional[torch.Tensor]=None, blocksize=64, compress_statistics=False, quant_storage=torch.uint8):
    return quantize_4bit(A, absmax, out, blocksize, compress_statistics, 'nf4', quant_storage)