import types
import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.name_scope import name_scope as base_name_scope
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.utils.naming import auto_name
def convert_keras_tensor_to_tf(x):
    if isinstance(x, KerasTensor):
        if x.sparse:
            return tf.compat.v1.sparse_placeholder(shape=x.shape, dtype=x.dtype)
        else:
            return tf.compat.v1.placeholder(shape=x.shape, dtype=x.dtype)
    if isinstance(x, types.FunctionType):

        def _fn(*x_args, **x_kwargs):
            out = x(*x_args, **x_kwargs)
            out = convert_keras_tensor_to_tf(out)
            return out
        return _fn
    return x