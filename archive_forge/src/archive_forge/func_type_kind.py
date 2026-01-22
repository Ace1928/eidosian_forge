from ctypes import c_int, c_bool, c_void_p, c_uint64
import enum
from llvmlite.binding import ffi
@property
def type_kind(self):
    """
        Returns the LLVMTypeKind enumeration of this type.
        """
    return TypeKind(ffi.lib.LLVMPY_GetTypeKind(self))