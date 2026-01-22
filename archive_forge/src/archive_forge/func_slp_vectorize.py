from ctypes import c_uint, c_bool
from llvmlite.binding import ffi
from llvmlite.binding import passmanagers
@slp_vectorize.setter
def slp_vectorize(self, enable=True):
    return ffi.lib.LLVMPY_PassManagerBuilderSetSLPVectorize(self, enable)