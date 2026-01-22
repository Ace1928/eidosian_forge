import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
def get_element_offset(self, ty, position):
    """
        Get byte offset of type's ty element at the given position
        """
    offset = ffi.lib.LLVMPY_OffsetOfElement(self, ty, position)
    if offset == -1:
        raise ValueError("Could not determined offset of {}th element of the type '{}'. Is it a structtype?".format(position, str(ty)))
    return offset