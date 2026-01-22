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
@tf_export.tf_export('experimental.numpy.vander', v1=[])
@np_utils.np_doc('vander')
def vander(x, N=None, increasing=False):
    x = asarray(x)
    x_shape = array_ops.shape(x)
    N = N or x_shape[0]
    N_temp = np_utils.get_static_value(N)
    if N_temp is not None:
        N = N_temp
        if N < 0:
            raise ValueError('N must be nonnegative')
    else:
        control_flow_assert.Assert(N >= 0, [N])
    rank = array_ops.rank(x)
    rank_temp = np_utils.get_static_value(rank)
    if rank_temp is not None:
        rank = rank_temp
        if rank != 1:
            raise ValueError('x must be a one-dimensional array')
    else:
        control_flow_assert.Assert(math_ops.equal(rank, 1), [rank])
    if increasing:
        start = 0
        limit = N
        delta = 1
    else:
        start = N - 1
        limit = -1
        delta = -1
    x = array_ops.expand_dims(x, -1)
    return math_ops.pow(x, math_ops.cast(math_ops.range(start, limit, delta), dtype=x.dtype))