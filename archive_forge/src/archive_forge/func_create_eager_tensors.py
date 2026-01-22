import json
import shutil
import tempfile
import unittest
import numpy as np
import tree
from keras.src import backend
from keras.src import ops
from keras.src import utils
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.global_state import clear_session
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.models import Model
from keras.src.utils import traceback_utils
from keras.src.utils.shape_utils import map_shape_structure
def create_eager_tensors(input_shape, dtype, sparse):
    from keras.src.backend import random
    if dtype not in ['float16', 'float32', 'float64', 'int16', 'int32', 'int64']:
        raise ValueError(f'dtype must be a standard float or int dtype. Received: dtype={dtype}')
    if sparse:
        if backend.backend() == 'tensorflow':
            import tensorflow as tf

            def create_fn(shape):
                rng = np.random.default_rng(0)
                x = (4 * rng.standard_normal(shape)).astype(dtype)
                x = np.multiply(x, rng.random(shape) < 0.7)
                return tf.sparse.from_dense(x)
        elif backend.backend() == 'jax':
            import jax.experimental.sparse as jax_sparse

            def create_fn(shape):
                rng = np.random.default_rng(0)
                x = (4 * rng.standard_normal(shape)).astype(dtype)
                x = np.multiply(x, rng.random(shape) < 0.7)
                return jax_sparse.BCOO.fromdense(x, n_batch=1)
        else:
            raise ValueError(f'Sparse is unsupported with backend {backend.backend()}')
    else:

        def create_fn(shape):
            return ops.cast(random.uniform(shape, dtype='float32') * 3, dtype=dtype)
    if isinstance(input_shape, dict):
        return {utils.removesuffix(k, '_shape'): create_fn(v) for k, v in input_shape.items()}
    return map_shape_structure(create_fn, input_shape)