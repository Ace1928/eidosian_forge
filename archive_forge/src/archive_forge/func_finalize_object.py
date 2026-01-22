import platform
from ctypes import (POINTER, c_char_p, c_bool, c_void_p,
from llvmlite.binding import ffi, targets, object_file
def finalize_object(self):
    """
        Make sure all modules owned by the execution engine are fully processed
        and "usable" for execution.
        """
    ffi.lib.LLVMPY_FinalizeObject(self)