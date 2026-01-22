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
@tf_export.tf_export('experimental.numpy.linspace', v1=[])
@np_utils.np_doc('linspace')
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=float, axis=0):
    if dtype:
        dtype = np_utils.result_type(dtype)
    start = np_array_ops.array(start, dtype=dtype)
    stop = np_array_ops.array(stop, dtype=dtype)
    if num < 0:
        raise ValueError(f'Argument `num` (number of samples) must be a non-negative integer. Received: num={num}')
    step = ops.convert_to_tensor(np.nan)
    if endpoint:
        result = math_ops.linspace(start, stop, num, axis=axis)
        if num > 1:
            step = (stop - start) / (num - 1)
    else:
        if num > 0:
            step = (stop - start) / num
        if num > 1:
            new_stop = math_ops.cast(stop, step.dtype) - step
            start = math_ops.cast(start, new_stop.dtype)
            result = math_ops.linspace(start, new_stop, num, axis=axis)
        else:
            result = math_ops.linspace(start, stop, num, axis=axis)
    if dtype:
        if dtype.is_integer:
            result = math_ops.floor(result)
        result = math_ops.cast(result, dtype)
    if retstep:
        return (result, step)
    else:
        return result