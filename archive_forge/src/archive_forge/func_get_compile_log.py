from ctypes import byref, c_char, c_char_p, c_int, c_size_t, c_void_p, POINTER
from enum import IntEnum
from numba.core import config
from numba.cuda.cudadrv.error import (NvrtcError, NvrtcCompilationError,
import functools
import os
import threading
import warnings
def get_compile_log(self, program):
    """
        Get the compile log as a Python string.
        """
    log_size = c_size_t()
    self.nvrtcGetProgramLogSize(program.handle, byref(log_size))
    log = (c_char * log_size.value)()
    self.nvrtcGetProgramLog(program.handle, log)
    return log.value.decode()