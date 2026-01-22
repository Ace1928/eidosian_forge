from ctypes import c_int, c_bool, c_void_p, c_uint64
import enum
from llvmlite.binding import ffi
@property
def is_struct(self):
    """
        Returns true if the type is a struct type.
        """
    return ffi.lib.LLVMPY_TypeIsStruct(self)