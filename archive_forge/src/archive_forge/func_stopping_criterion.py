import collections
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def stopping_criterion(i, state):
    return math_ops.logical_and(i < max_iter, math_ops.reduce_any(linalg.norm(state.r, axis=-1) > tol))