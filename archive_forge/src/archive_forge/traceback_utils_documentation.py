import os
import sys
import threading
import traceback
import types
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
Decorator to filter out TF-internal stack trace frames in exceptions.

  Raw TensorFlow stack traces involve many internal frames, which can be
  challenging to read through, while not being actionable for end users.
  By default, TensorFlow filters internal frames in most exceptions that it
  raises, to keep stack traces short, readable, and focused on what's
  actionable for end users (their own code).

  Arguments:
    fn: The function or method to decorate. Any exception raised within the
      function will be reraised with its internal stack trace frames filtered
      out.

  Returns:
    Decorated function or method.
  