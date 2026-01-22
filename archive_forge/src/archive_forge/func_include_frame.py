import os
import sys
import threading
import traceback
import types
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
def include_frame(fname):
    for exclusion in _EXCLUDED_PATHS:
        if exclusion in fname:
            return False
    return True