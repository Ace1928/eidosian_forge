import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
def get_pointee_abi_alignment(self, ty):
    """
        Get minimum ABI alignment of pointee type of LLVM pointer type *ty*.
        """
    size = ffi.lib.LLVMPY_ABIAlignmentOfElementType(self, ty)
    if size == -1:
        raise RuntimeError('Not a pointer type: %s' % (ty,))
    return size