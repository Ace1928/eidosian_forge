import contextlib
import sys
import threading
import time
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def raise_requested_exception(self):
    """If an exception has been passed to `request_stop`, this raises it."""
    with self._lock:
        if self._exc_info_to_raise:
            _, ex_instance, _ = self._exc_info_to_raise
            raise ex_instance