import atexit
import struct
import warnings
import pyglet
from . import com
from . import constants
from .types import *
def win32_errcheck(result, func, args):
    last_err = ctypes.get_last_error()
    if last_err != 0:
        for entry in traceback.format_list(traceback.extract_stack()[:-1]):
            _log_win32.write(entry)
        print(f'[Result {result}] Error #{last_err} - {ctypes.FormatError(last_err)}', file=_log_win32)
    return args