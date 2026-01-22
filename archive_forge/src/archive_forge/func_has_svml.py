import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
def has_svml():
    """
    Returns True if SVML was enabled at FFI support compile time.
    """
    if ffi.lib.LLVMPY_HasSVMLSupport() == 0:
        return False
    else:
        return True