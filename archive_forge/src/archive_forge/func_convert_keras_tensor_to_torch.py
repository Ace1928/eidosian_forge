import contextlib
import os
import ml_dtypes
import numpy as np
import torch
import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.config import floatx
from keras.src.utils.nest import pack_sequence_as
def convert_keras_tensor_to_torch(x, fill_value=None):
    """Convert `KerasTensor`s to `torch.Tensor`s."""
    if isinstance(x, KerasTensor):
        shape = list(x.shape)
        if fill_value:
            for i, e in enumerate(shape):
                if e is None:
                    shape[i] = fill_value
        return torch.ones(size=shape, dtype=TORCH_DTYPES[x.dtype], device=get_device())
    return x