import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_uint8, c_uint64, c_size_t
from llvmlite.binding import ffi, targets
def create_lljit_compiler(target_machine=None, *, use_jit_link=False, suppress_errors=False):
    """
    Create an LLJIT instance
    """
    with ffi.OutputString() as outerr:
        lljit = ffi.lib.LLVMPY_CreateLLJITCompiler(target_machine, suppress_errors, use_jit_link, outerr)
        if not lljit:
            raise RuntimeError(str(outerr))
    return LLJIT(lljit)