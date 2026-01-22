from ctypes import byref, c_char, c_char_p, c_int, c_size_t, c_void_p, POINTER
from enum import IntEnum
from numba.core import config
from numba.cuda.cudadrv.error import (NvrtcError, NvrtcCompilationError,
import functools
import os
import threading
import warnings
def destroy_program(self, program):
    """
        Destroy an NVRTC program.
        """
    self.nvrtcDestroyProgram(byref(program.handle))