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
@tf_export.tf_export('experimental.numpy.geomspace', v1=[])
@np_utils.np_doc('geomspace')
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    dtype = dtypes.as_dtype(dtype) if dtype else np_utils.result_type(start, stop, float(num), np_array_ops.zeros((), dtype))
    computation_dtype = np.promote_types(dtype.as_numpy_dtype, np.float32)
    start = np_array_ops.asarray(start, dtype=computation_dtype)
    stop = np_array_ops.asarray(stop, dtype=computation_dtype)
    start_sign = 1 - np_array_ops.sign(np_array_ops.real(start))
    stop_sign = 1 - np_array_ops.sign(np_array_ops.real(stop))
    signflip = 1 - start_sign * stop_sign // 2
    res = signflip * logspace(log10(signflip * start), log10(signflip * stop), num, endpoint=endpoint, base=10.0, dtype=computation_dtype, axis=0)
    if axis != 0:
        res = np_array_ops.moveaxis(res, 0, axis)
    return math_ops.cast(res, dtype)