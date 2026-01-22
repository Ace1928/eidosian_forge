import ctypes
import enum
import os
import platform
import sys
import numpy as np
@_tf_export('lite.experimental.load_delegate')
def load_delegate(library, options=None):
    """Returns loaded Delegate object.

  Example usage:

  ```
  import tensorflow as tf

  try:
    delegate = tf.lite.experimental.load_delegate('delegate.so')
  except ValueError:
    // Fallback to CPU

  if delegate:
    interpreter = tf.lite.Interpreter(
        model_path='model.tflite',
        experimental_delegates=[delegate])
  else:
    interpreter = tf.lite.Interpreter(model_path='model.tflite')
  ```

  This is typically used to leverage EdgeTPU for running TensorFlow Lite models.
  For more information see: https://coral.ai/docs/edgetpu/tflite-python/

  Args:
    library: Name of shared library containing the
      [TfLiteDelegate](https://www.tensorflow.org/lite/performance/delegates).
    options: Dictionary of options that are required to load the delegate. All
      keys and values in the dictionary should be convertible to str. Consult
      the documentation of the specific delegate for required and legal options.
      (default None)

  Returns:
    Delegate object.

  Raises:
    ValueError: Delegate failed to load.
    RuntimeError: If delegate loading is used on unsupported platform.
  """
    try:
        delegate = Delegate(library, options)
    except ValueError as e:
        raise ValueError('Failed to load delegate from {}\n{}'.format(library, str(e)))
    return delegate