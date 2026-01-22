import functools
from tensorflow.lite.python import convert
from tensorflow.lite.python import lite
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.python.util.tf_export import tf_export as _tf_export
@_tf_export('lite.experimental.authoring.compatible')
def compatible(target=None, converter_target_spec=None, **kwargs):
    """Wraps `tf.function` into a callable function with TFLite compatibility checking.

  Example:

  ```python
  @tf.lite.experimental.authoring.compatible
  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.float32)
  ])
  def f(x):
      return tf.cosh(x)

  result = f(tf.constant([0.0]))
  # COMPATIBILITY WARNING: op 'tf.Cosh' require(s) "Select TF Ops" for model
  # conversion for TensorFlow Lite.
  # Op: tf.Cosh
  #   - tensorflow/python/framework/op_def_library.py:748
  #   - tensorflow/python/ops/gen_math_ops.py:2458
  #   - <stdin>:6
  ```

  WARNING: Experimental interface, subject to change.

  Args:
    target: A `tf.function` to decorate.
    converter_target_spec : target_spec of TFLite converter parameter.
    **kwargs: The keyword arguments of the decorator class _Compatible.

  Returns:
     A callable object of `tf.lite.experimental.authoring._Compatible`.
  """
    if target is None:

        def wrapper(target):
            return _Compatible(target, converter_target_spec, **kwargs)
        return wrapper
    else:
        return _Compatible(target, converter_target_spec, **kwargs)