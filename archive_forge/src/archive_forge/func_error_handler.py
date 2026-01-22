import os
import sys
import threading
import traceback
import types
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
def error_handler(*args, **kwargs):
    try:
        if not is_traceback_filtering_enabled():
            return fn(*args, **kwargs)
    except NameError:
        return fn(*args, **kwargs)
    filtered_tb = None
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        filtered_tb = _process_traceback_frames(e.__traceback__)
        raise e.with_traceback(filtered_tb) from None
    finally:
        del filtered_tb