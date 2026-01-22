from ctypes import byref, c_char, c_char_p, c_int, c_size_t, c_void_p, POINTER
from enum import IntEnum
from numba.core import config
from numba.cuda.cudadrv.error import (NvrtcError, NvrtcCompilationError,
import functools
import os
import threading
import warnings
def get_ptx(self, program):
    """
        Get the compiled PTX as a Python string.
        """
    ptx_size = c_size_t()
    self.nvrtcGetPTXSize(program.handle, byref(ptx_size))
    ptx = (c_char * ptx_size.value)()
    self.nvrtcGetPTX(program.handle, ptx)
    return ptx.value.decode()