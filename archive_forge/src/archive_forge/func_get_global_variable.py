from ctypes import (c_char_p, byref, POINTER, c_bool, create_string_buffer,
from llvmlite.binding import ffi
from llvmlite.binding.linker import link_modules
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.value import ValueRef, TypeRef
from llvmlite.binding.context import get_global_context
def get_global_variable(self, name):
    """
        Get a ValueRef pointing to the global variable named *name*.
        NameError is raised if the symbol isn't found.
        """
    p = ffi.lib.LLVMPY_GetNamedGlobalVariable(self, _encode_string(name))
    if not p:
        raise NameError(name)
    return ValueRef(p, 'global', dict(module=self))