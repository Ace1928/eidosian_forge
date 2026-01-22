from ctypes import (c_char_p, byref, POINTER, c_bool, create_string_buffer,
from llvmlite.binding import ffi
from llvmlite.binding.linker import link_modules
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.value import ValueRef, TypeRef
from llvmlite.binding.context import get_global_context
def link_in(self, other, preserve=False):
    """
        Link the *other* module into this one.  The *other* module will
        be destroyed unless *preserve* is true.
        """
    if preserve:
        other = other.clone()
    link_modules(self, other)