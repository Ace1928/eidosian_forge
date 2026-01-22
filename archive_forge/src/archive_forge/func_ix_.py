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
@tf_export.tf_export('experimental.numpy.ix_', v1=[])
@np_utils.np_doc('ix_')
def ix_(*args):
    n = len(args)
    output = []
    for i, a in enumerate(args):
        a = asarray(a)
        a_rank = array_ops.rank(a)
        a_rank_temp = np_utils.get_static_value(a_rank)
        if a_rank_temp is not None:
            a_rank = a_rank_temp
            if a_rank != 1:
                raise ValueError('Arguments must be 1-d, got arg {} of rank {}'.format(i, a_rank))
        else:
            control_flow_assert.Assert(math_ops.equal(a_rank, 1), [a_rank])
        new_shape = [1] * n
        new_shape[i] = -1
        dtype = a.dtype
        if dtype == dtypes.bool:
            output.append(array_ops.reshape(nonzero(a)[0], new_shape))
        elif dtype.is_integer:
            output.append(array_ops.reshape(a, new_shape))
        else:
            raise ValueError('Only integer and bool dtypes are supported, got {}'.format(dtype))
    return output