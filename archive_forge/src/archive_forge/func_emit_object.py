import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
def emit_object(self, module):
    """
        Represent the module as a code object, suitable for use with
        the platform's linker.  Returns a byte string.
        """
    return self._emit_to_memory(module, use_object=True)