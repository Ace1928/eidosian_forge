import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('ensure_shape')
@dispatch.add_dispatch_support
def ensure_shape(x, shape, name=None):
    """Updates the shape of a tensor and checks at runtime that the shape holds.

  When executed, this operation asserts that the input tensor `x`'s shape
  is compatible with the `shape` argument.
  See `tf.TensorShape.is_compatible_with` for details.

  >>> x = tf.constant([[1, 2, 3],
  ...                  [4, 5, 6]])
  >>> x = tf.ensure_shape(x, [2, 3])

  Use `None` for unknown dimensions:

  >>> x = tf.ensure_shape(x, [None, 3])
  >>> x = tf.ensure_shape(x, [2, None])

  If the tensor's shape is not compatible with the `shape` argument, an error
  is raised:

  >>> x = tf.ensure_shape(x, [5])
  Traceback (most recent call last):
  ...
  tf.errors.InvalidArgumentError: Shape of tensor dummy_input [3] is not
    compatible with expected shape [5]. [Op:EnsureShape]

  During graph construction (typically tracing a `tf.function`),
  `tf.ensure_shape` updates the static-shape of the **result** tensor by
  merging the two shapes. See `tf.TensorShape.merge_with` for details.

  This is most useful when **you** know a shape that can't be determined
  statically by TensorFlow.

  The following trivial `tf.function` prints the input tensor's
  static-shape before and after `ensure_shape` is applied.

  >>> @tf.function
  ... def f(tensor):
  ...   print("Static-shape before:", tensor.shape)
  ...   tensor = tf.ensure_shape(tensor, [None, 3])
  ...   print("Static-shape after:", tensor.shape)
  ...   return tensor

  This lets you see the effect of `tf.ensure_shape` when the function is traced:
  >>> cf = f.get_concrete_function(tf.TensorSpec([None, None]))
  Static-shape before: (None, None)
  Static-shape after: (None, 3)

  >>> cf(tf.zeros([3, 3])) # Passes
  >>> cf(tf.constant([1, 2, 3])) # fails
  Traceback (most recent call last):
  ...
  InvalidArgumentError:  Shape of tensor x [3] is not compatible with expected shape [3,3].

  The above example raises `tf.errors.InvalidArgumentError`, because `x`'s
  shape, `(3,)`, is not compatible with the `shape` argument, `(None, 3)`

  Inside a `tf.function` or `v1.Graph` context it checks both the buildtime and
  runtime shapes. This is stricter than `tf.Tensor.set_shape` which only
  checks the buildtime shape.

  Note: This differs from `tf.Tensor.set_shape` in that it sets the static shape
  of the resulting tensor and enforces it at runtime, raising an error if the
  tensor's runtime shape is incompatible with the specified shape.
  `tf.Tensor.set_shape` sets the static shape of the tensor without enforcing it
  at runtime, which may result in inconsistencies between the statically-known
  shape of tensors and the runtime value of tensors.

  For example, of loading images of a known size:

  >>> @tf.function
  ... def decode_image(png):
  ...   image = tf.image.decode_png(png, channels=3)
  ...   # the `print` executes during tracing.
  ...   print("Initial shape: ", image.shape)
  ...   image = tf.ensure_shape(image,[28, 28, 3])
  ...   print("Final shape: ", image.shape)
  ...   return image

  When tracing a function, no ops are being executed, shapes may be unknown.
  See the [Concrete Functions Guide](https://www.tensorflow.org/guide/concrete_function)
  for details.

  >>> concrete_decode = decode_image.get_concrete_function(
  ...     tf.TensorSpec([], dtype=tf.string))
  Initial shape:  (None, None, 3)
  Final shape:  (28, 28, 3)

  >>> image = tf.random.uniform(maxval=255, shape=[28, 28, 3], dtype=tf.int32)
  >>> image = tf.cast(image,tf.uint8)
  >>> png = tf.image.encode_png(image)
  >>> image2 = concrete_decode(png)
  >>> print(image2.shape)
  (28, 28, 3)

  >>> image = tf.concat([image,image], axis=0)
  >>> print(image.shape)
  (56, 28, 3)
  >>> png = tf.image.encode_png(image)
  >>> image2 = concrete_decode(png)
  Traceback (most recent call last):
  ...
  tf.errors.InvalidArgumentError:  Shape of tensor DecodePng [56,28,3] is not
    compatible with expected shape [28,28,3].

  Caution: if you don't use the result of `tf.ensure_shape` the check may not
  run.

  >>> @tf.function
  ... def bad_decode_image(png):
  ...   image = tf.image.decode_png(png, channels=3)
  ...   # the `print` executes during tracing.
  ...   print("Initial shape: ", image.shape)
  ...   # BAD: forgot to use the returned tensor.
  ...   tf.ensure_shape(image,[28, 28, 3])
  ...   print("Final shape: ", image.shape)
  ...   return image

  >>> image = bad_decode_image(png)
  Initial shape:  (None, None, 3)
  Final shape:  (None, None, 3)
  >>> print(image.shape)
  (56, 28, 3)

  Args:
    x: A `Tensor`.
    shape: A `TensorShape` representing the shape of this tensor, a
      `TensorShapeProto`, a list, a tuple, or None.
    name: A name for this operation (optional). Defaults to "EnsureShape".

  Returns:
    A `Tensor`. Has the same type and contents as `x`.

  Raises:
    tf.errors.InvalidArgumentError: If `shape` is incompatible with the shape
    of `x`.
  """
    if not isinstance(shape, tensor_shape.TensorShape):
        shape = tensor_shape.TensorShape(shape)
    return array_ops.ensure_shape(x, shape, name=name)