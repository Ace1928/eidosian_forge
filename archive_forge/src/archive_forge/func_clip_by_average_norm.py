from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(date=None, instructions='clip_by_average_norm is deprecated in TensorFlow 2.0. Please use clip_by_norm(t, clip_norm * tf.cast(tf.size(t), tf.float32), name) instead.')
@tf_export(v1=['clip_by_average_norm'])
@dispatch.add_dispatch_support
def clip_by_average_norm(t, clip_norm, name=None):
    """Clips tensor values to a maximum average L2-norm.

  Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
  normalizes `t` so that its average L2-norm is less than or equal to
  `clip_norm`. Specifically, if the average L2-norm is already less than or
  equal to `clip_norm`, then `t` is not modified. If the average L2-norm is
  greater than `clip_norm`, then this operation returns a tensor of the same
  type and shape as `t` with its values set to:

  `t * clip_norm / l2norm_avg(t)`

  In this case, the average L2-norm of the output tensor is `clip_norm`.

  This operation is typically used to clip gradients before applying them with
  an optimizer.

  Args:
    t: A `Tensor`.
    clip_norm: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
    name: A name for the operation (optional).

  Returns:
    A clipped `Tensor`.
  """
    with ops.name_scope(name, 'clip_by_average_norm', [t, clip_norm]) as name:
        t = ops.convert_to_tensor(t, name='t')
        n_element = math_ops.cast(array_ops.size(t), dtypes.float32)
        l2norm_inv = math_ops.rsqrt(math_ops.reduce_sum(t * t, math_ops.range(array_ops.rank(t))))
        tclip = array_ops.identity(t * clip_norm * math_ops.minimum(l2norm_inv * n_element, constant_op.constant(1.0) / clip_norm), name=name)
    return tclip