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
def sparse_sparse_matmul(a, b):
    dtype = a.values.dtype
    a_csr = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(a.indices, a.values, a.dense_shape)
    b_csr = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(b.indices, b.values, b.dense_shape)
    result_csr = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(a_csr, b_csr, dtype)
    res = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(result_csr, dtype)
    return tf.SparseTensor(res.indices, res.values, res.dense_shape)