import builtins
import collections
import math
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.experimental import numpy as tfnp
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.tensorflow import sparse
from keras.src.backend.tensorflow.core import convert_to_tensor
def nan_to_num(x):
    x = convert_to_tensor(x)
    dtype = x.dtype
    dtype_as_dtype = tf.as_dtype(dtype)
    if dtype_as_dtype.is_integer or not dtype_as_dtype.is_numeric:
        return x
    x = tf.where(tf.math.is_nan(x), tf.constant(0, dtype), x)
    x = tf.where(tf.math.is_inf(x) & (x > 0), tf.constant(dtype.max, dtype), x)
    x = tf.where(tf.math.is_inf(x) & (x < 0), tf.constant(dtype.min, dtype), x)
    return x