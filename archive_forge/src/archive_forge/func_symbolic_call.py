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
def symbolic_call(fn, args, kwargs, fill_value):
    """Call `fn` to infer output shape and dtype."""
    try:
        with device_scope('meta'):
            meta_args, meta_kwargs = tree.map_structure(lambda x: convert_keras_tensor_to_torch(x, fill_value), (args, kwargs))
            return fn(*meta_args, **meta_kwargs)
    except:
        with device_scope(DEFAULT_DEVICE):
            eager_args, eager_kwargs = tree.map_structure(lambda x: convert_keras_tensor_to_torch(x, fill_value), (args, kwargs))
            return fn(*eager_args, **eager_kwargs)