import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
@classmethod
def from_triple(cls, triple):
    """
        Create a Target instance for the given triple (a string).
        """
    with ffi.OutputString() as outerr:
        target = ffi.lib.LLVMPY_GetTargetFromTriple(triple.encode('utf8'), outerr)
        if not target:
            raise RuntimeError(str(outerr))
        target = cls(target)
        target._triple = triple
        return target