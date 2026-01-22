import builtins
import enum
import functools
import math
import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
@tf_export.tf_export('experimental.numpy.var', v1=[])
@np_utils.np_doc('var')
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=None):
    if dtype:
        working_dtype = np_utils.result_type(a, dtype)
    else:
        working_dtype = None
    if out is not None:
        raise ValueError('Setting out is not supported.')
    if ddof != 0:

        def reduce_fn(input_tensor, axis, keepdims):
            means = math_ops.reduce_mean(input_tensor, axis=axis, keepdims=True)
            centered = input_tensor - means
            if input_tensor.dtype in (dtypes.complex64, dtypes.complex128):
                centered = math_ops.cast(math_ops.real(centered * math_ops.conj(centered)), input_tensor.dtype)
            else:
                centered = math_ops.square(centered)
            squared_deviations = math_ops.reduce_sum(centered, axis=axis, keepdims=keepdims)
            if axis is None:
                n = array_ops.size(input_tensor)
            else:
                if axis < 0:
                    axis += array_ops.rank(input_tensor)
                n = math_ops.reduce_prod(array_ops.gather(array_ops.shape(input_tensor), axis))
            n = math_ops.cast(n - ddof, input_tensor.dtype)
            return math_ops.cast(math_ops.divide(squared_deviations, n), dtype)
    else:
        reduce_fn = math_ops.reduce_variance
    result = _reduce(reduce_fn, a, axis=axis, dtype=working_dtype, keepdims=keepdims, promote_int=_TO_FLOAT)
    if dtype:
        result = math_ops.cast(result, dtype)
    return result