from ctypes import c_uint, c_bool
from llvmlite.binding import ffi
from llvmlite.binding import passmanagers
@disable_unroll_loops.setter
def disable_unroll_loops(self, disable=True):
    ffi.lib.LLVMPY_PassManagerBuilderSetDisableUnrollLoops(self, disable)