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
@tf_export.tf_export('experimental.numpy.hstack', v1=[])
@np_utils.np_doc('hstack')
def hstack(tup):
    arrays = [atleast_1d(a) for a in tup]
    arrays = _promote_dtype(*arrays)
    unwrapped_arrays = [a if isinstance(a, np_arrays.ndarray) else a for a in arrays]
    rank = array_ops.rank(unwrapped_arrays[0])
    return np_utils.cond(math_ops.equal(rank, 1), lambda: array_ops.concat(unwrapped_arrays, axis=0), lambda: array_ops.concat(unwrapped_arrays, axis=1))