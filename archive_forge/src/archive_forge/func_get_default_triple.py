import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
def get_default_triple():
    """
    Return the default target triple LLVM is configured to produce code for.
    """
    with ffi.OutputString() as out:
        ffi.lib.LLVMPY_GetDefaultTargetTriple(out)
        return str(out)