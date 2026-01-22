import os
import sys
import threading
import traceback
import types
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
@tf_export('debugging.disable_traceback_filtering')
def disable_traceback_filtering():
    """Disable filtering out TensorFlow-internal frames in exception stack traces.

  Raw TensorFlow stack traces involve many internal frames, which can be
  challenging to read through, while not being actionable for end users.
  By default, TensorFlow filters internal frames in most exceptions that it
  raises, to keep stack traces short, readable, and focused on what's
  actionable for end users (their own code).

  Calling `tf.debugging.disable_traceback_filtering` disables this filtering
  mechanism, meaning that TensorFlow exceptions stack traces will include
  all frames, in particular TensorFlow-internal ones.

  **If you are debugging a TensorFlow-internal issue, you need to call
  `tf.debugging.disable_traceback_filtering`**.
  To re-enable traceback filtering afterwards, you can call
  `tf.debugging.enable_traceback_filtering()`.
  """
    global _ENABLE_TRACEBACK_FILTERING
    _ENABLE_TRACEBACK_FILTERING.value = False