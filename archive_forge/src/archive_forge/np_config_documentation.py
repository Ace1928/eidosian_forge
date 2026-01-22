from tensorflow.python.framework import ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.util import tf_export
Enable NumPy behavior on Tensors.

  Enabling NumPy behavior has three effects:
  * It adds to `tf.Tensor` some common NumPy methods such as `T`,
    `reshape` and `ravel`.
  * It changes dtype promotion in `tf.Tensor` operators to be
    compatible with NumPy. For example,
    `tf.ones([], tf.int32) + tf.ones([], tf.float32)` used to throw a
    "dtype incompatible" error, but after this it will return a
    float64 tensor (obeying NumPy's promotion rules).
  * It enhances `tf.Tensor`'s indexing capability to be on par with
    [NumPy's](https://numpy.org/doc/stable/reference/arrays.indexing.html).

  Args:
    prefer_float32: Controls whether dtype inference will use float32 for Python
      floats, or float64 (the default and the NumPy-compatible behavior).
    dtype_conversion_mode: a string that specifies promotion mode. This string
      corresponds to a PromoMode Enum and can be 'off', 'legacy', 'safe', or
      'all'. 'safe' or 'all' mode enables the auto dtype conversion semantics.
  