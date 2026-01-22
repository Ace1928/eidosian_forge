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
def digitize(x, bins):
    x = convert_to_tensor(x)
    bins = list(bins)
    bins = tf.nest.map_structure(lambda x: float(x), bins)
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype in ('bool', 'int8', 'int16', 'uint8', 'uint16'):
        x = tf.cast(x, 'int32')
    elif ori_dtype == 'uint32':
        x = tf.cast(x, 'int64')
    elif ori_dtype in ('bfloat16', 'float16'):
        x = tf.cast(x, 'float32')
    if isinstance(x, tf.RaggedTensor):
        return tf.ragged.map_flat_values(lambda y: tf.raw_ops.Bucketize(input=y, boundaries=bins), x)
    elif isinstance(x, tf.SparseTensor):
        output = tf.SparseTensor(indices=tf.identity(x.indices), values=tf.raw_ops.Bucketize(input=x.values, boundaries=bins), dense_shape=tf.identity(x.dense_shape))
        output.set_shape(x.shape)
        return output
    return tf.raw_ops.Bucketize(input=x, boundaries=bins)