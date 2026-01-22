import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
def set_asm_verbosity(self, verbose):
    """
        Set whether this target machine will emit assembly with human-readable
        comments describing control flow, debug information, and so on.
        """
    ffi.lib.LLVMPY_SetTargetMachineAsmVerbosity(self, verbose)