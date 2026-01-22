import functools
import hashlib
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util import tf_inspect
def tridiag(below=None, diag=None, above=None, name=None):
    """Creates a matrix with values set above, below, and on the diagonal.

  Example:

  ```python
  tridiag(below=[1., 2., 3.],
          diag=[4., 5., 6., 7.],
          above=[8., 9., 10.])
  # ==> array([[  4.,   8.,   0.,   0.],
  #            [  1.,   5.,   9.,   0.],
  #            [  0.,   2.,   6.,  10.],
  #            [  0.,   0.,   3.,   7.]], dtype=float32)
  ```

  Warning: This Op is intended for convenience, not efficiency.

  Args:
    below: `Tensor` of shape `[B1, ..., Bb, d-1]` corresponding to the below
      diagonal part. `None` is logically equivalent to `below = 0`.
    diag: `Tensor` of shape `[B1, ..., Bb, d]` corresponding to the diagonal
      part.  `None` is logically equivalent to `diag = 0`.
    above: `Tensor` of shape `[B1, ..., Bb, d-1]` corresponding to the above
      diagonal part.  `None` is logically equivalent to `above = 0`.
    name: Python `str`. The name to give this op.

  Returns:
    tridiag: `Tensor` with values set above, below and on the diagonal.

  Raises:
    ValueError: if all inputs are `None`.
  """

    def _pad(x):
        """Prepends and appends a zero to every vector in a batch of vectors."""
        shape = array_ops.concat([array_ops.shape(x)[:-1], [1]], axis=0)
        z = array_ops.zeros(shape, dtype=x.dtype)
        return array_ops.concat([z, x, z], axis=-1)

    def _add(*x):
        """Adds list of Tensors, ignoring `None`."""
        s = None
        for y in x:
            if y is None:
                continue
            elif s is None:
                s = y
            else:
                s += y
        if s is None:
            raise ValueError('Must specify at least one of `below`, `diag`, `above`.')
        return s
    with ops.name_scope(name, 'tridiag', [below, diag, above]):
        if below is not None:
            below = ops.convert_to_tensor(below, name='below')
            below = array_ops.matrix_diag(_pad(below))[..., :-1, 1:]
        if diag is not None:
            diag = ops.convert_to_tensor(diag, name='diag')
            diag = array_ops.matrix_diag(diag)
        if above is not None:
            above = ops.convert_to_tensor(above, name='above')
            above = array_ops.matrix_diag(_pad(above))[..., 1:, :-1]
        return _add(below, diag, above)