import numbers
import sys
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export
@tf_export.tf_export('experimental.numpy.nanmean', v1=[])
@np_utils.np_doc('nanmean')
def nanmean(a, axis=None, dtype=None, keepdims=None):
    a = np_array_ops.array(a)
    if np.issubdtype(a.dtype.as_numpy_dtype, np.bool_) or np.issubdtype(a.dtype.as_numpy_dtype, np.integer):
        return np_array_ops.mean(a, axis=axis, dtype=dtype, keepdims=keepdims)
    nan_mask = logical_not(isnan(a))
    if dtype is None:
        dtype = a.dtype.as_numpy_dtype
    normalizer = np_array_ops.sum(nan_mask, axis=axis, dtype=dtype, keepdims=keepdims)
    return nansum(a, axis=axis, dtype=dtype, keepdims=keepdims) / normalizer