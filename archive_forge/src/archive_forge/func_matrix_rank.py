import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('linalg.matrix_rank')
@dispatch.add_dispatch_support
def matrix_rank(a, tol=None, validate_args=False, name=None):
    """Compute the matrix rank of one or more matrices.

  Args:
    a: (Batch of) `float`-like matrix-shaped `Tensor`(s) which are to be
      pseudo-inverted.
    tol: Threshold below which the singular value is counted as 'zero'.
      Default value: `None` (i.e., `eps * max(rows, cols) * max(singular_val)`).
    validate_args: When `True`, additional assertions might be embedded in the
      graph.
      Default value: `False` (i.e., no graph assertions are added).
    name: Python `str` prefixed to ops created by this function.
      Default value: 'matrix_rank'.

  Returns:
    matrix_rank: (Batch of) `int32` scalars representing the number of non-zero
      singular values.
  """
    with ops.name_scope(name or 'matrix_rank'):
        a = ops.convert_to_tensor(a, dtype_hint=dtypes.float32, name='a')
        assertions = _maybe_validate_matrix(a, validate_args)
        if assertions:
            with ops.control_dependencies(assertions):
                a = array_ops.identity(a)
        s = svd(a, compute_uv=False)
        if tol is None:
            if a.shape[-2:].is_fully_defined():
                m = np.max(a.shape[-2:].as_list())
            else:
                m = math_ops.reduce_max(array_ops.shape(a)[-2:])
            eps = np.finfo(a.dtype.as_numpy_dtype).eps
            tol = eps * math_ops.cast(m, a.dtype) * math_ops.reduce_max(s, axis=-1, keepdims=True)
        return math_ops.reduce_sum(math_ops.cast(s > tol, dtypes.int32), axis=-1)