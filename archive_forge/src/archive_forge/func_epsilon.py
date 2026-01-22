from tensorflow.python.util import dispatch
@dispatch.add_dispatch_support
def epsilon():
    """Returns the value of the fuzz factor used in numeric expressions.

  Returns:
      A float.

  Example:
  >>> tf.keras.backend.epsilon()
  1e-07
  """
    return _EPSILON