import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
@tf_export('boolean_mask', v1=[])
@dispatch.add_dispatch_support
def boolean_mask_v2(tensor, mask, axis=None, name='boolean_mask'):
    """Apply boolean mask to tensor.

  Numpy equivalent is `tensor[mask]`.

  In general, `0 < dim(mask) = K <= dim(tensor)`, and `mask`'s shape must match
  the first K dimensions of `tensor`'s shape.  We then have:
    `boolean_mask(tensor, mask)[i, j1,...,jd] = tensor[i1,...,iK,j1,...,jd]`
  where `(i1,...,iK)` is the ith `True` entry of `mask` (row-major order).
  The `axis` could be used with `mask` to indicate the axis to mask from.
  In that case, `axis + dim(mask) <= dim(tensor)` and `mask`'s shape must match
  the first `axis + dim(mask)` dimensions of `tensor`'s shape.

  See also: `tf.ragged.boolean_mask`, which can be applied to both dense and
  ragged tensors, and can be used if you need to preserve the masked dimensions
  of `tensor` (rather than flattening them, as `tf.boolean_mask` does).

  Examples:

  >>> tensor = [0, 1, 2, 3]  # 1-D example
  >>> mask = np.array([True, False, True, False])
  >>> tf.boolean_mask(tensor, mask)
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([0, 2], dtype=int32)>

  >>> tensor = [[1, 2], [3, 4], [5, 6]] # 2-D example
  >>> mask = np.array([True, False, True])
  >>> tf.boolean_mask(tensor, mask)
  <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
  array([[1, 2],
         [5, 6]], dtype=int32)>

  Args:
    tensor:  N-D Tensor.
    mask:  K-D boolean Tensor, K <= N and K must be known statically.
    axis:  A 0-D int Tensor representing the axis in `tensor` to mask from. By
      default, axis is 0 which will mask from the first dimension. Otherwise K +
      axis <= N.
    name:  A name for this operation (optional).

  Returns:
    (N-K+1)-dimensional tensor populated by entries in `tensor` corresponding
    to `True` values in `mask`.

  Raises:
    ValueError:  If shapes do not conform.

  Examples:

  ```python
  # 2-D example
  tensor = [[1, 2], [3, 4], [5, 6]]
  mask = np.array([True, False, True])
  boolean_mask(tensor, mask)  # [[1, 2], [5, 6]]
  ```
  """
    return boolean_mask(tensor, mask, name, axis)