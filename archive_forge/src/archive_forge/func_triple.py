import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
@property
def triple(self):
    with ffi.OutputString() as out:
        ffi.lib.LLVMPY_GetTargetMachineTriple(self, out)
        return str(out)