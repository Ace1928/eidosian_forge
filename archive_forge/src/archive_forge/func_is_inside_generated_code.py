import contextlib
import imp
import inspect
import io
import sys
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import api
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
def is_inside_generated_code():
    """Tests whether the caller is generated code. Implementation-specific."""
    frame = inspect.currentframe()
    try:
        frame = frame.f_back
        internal_stack_functions = ('converted_call', '_call_unconverted')
        while frame is not None and frame.f_code.co_name in internal_stack_functions:
            frame = frame.f_back
        if frame is None:
            return False
        return 'ag__' in frame.f_locals
    finally:
        del frame