from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import sys
import threading
import time
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.tools import analytics
@contextlib.contextmanager
def catch_errors(self, source, session=None):
    """Context manager to report any errors within a block."""
    try:
        yield
    except Exception:
        self.record_error(source, sys.exc_info(), session)