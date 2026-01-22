import os
import sys
import threading
import traceback
import types
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
@tf_export('debugging.enable_traceback_filtering')
def enable_traceback_filtering():
    """Enable filtering out TensorFlow-internal frames in exception stack traces.

  Raw TensorFlow stack traces involve many internal frames, which can be
  challenging to read through, while not being actionable for end users.
  By default, TensorFlow filters internal frames in most exceptions that it
  raises, to keep stack traces short, readable, and focused on what's
  actionable for end users (their own code).

  If you have previously disabled traceback filtering via
  `tf.debugging.disable_traceback_filtering()`, you can re-enable it via
  `tf.debugging.enable_traceback_filtering()`.

  Raises:
    RuntimeError: If Python version is not at least 3.7.
  """
    if sys.version_info.major != 3 or sys.version_info.minor < 7:
        raise RuntimeError(f'Traceback filtering is only available with Python 3.7 or higher. This Python version: {sys.version}')
    global _ENABLE_TRACEBACK_FILTERING
    _ENABLE_TRACEBACK_FILTERING.value = True