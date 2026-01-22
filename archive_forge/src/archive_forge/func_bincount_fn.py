import builtins
import math
import torch
from keras.src.backend import KerasTensor
from keras.src.backend import config
from keras.src.backend.common import dtypes
from keras.src.backend.common.variables import standardize_dtype
from keras.src.backend.torch.core import cast
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.core import get_device
from keras.src.backend.torch.core import is_tensor
from keras.src.backend.torch.core import to_torch_dtype
def bincount_fn(arr_w):
    return torch.bincount(arr_w[0], weights=arr_w[1], minlength=minlength)