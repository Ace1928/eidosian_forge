from ctypes import (c_char_p, byref, POINTER, c_bool, create_string_buffer,
from llvmlite.binding import ffi
from llvmlite.binding.linker import link_modules
from llvmlite.binding.common import _decode_string, _encode_string
from llvmlite.binding.value import ValueRef, TypeRef
from llvmlite.binding.context import get_global_context
def parse_assembly(llvmir, context=None):
    """
    Create Module from a LLVM IR string
    """
    if context is None:
        context = get_global_context()
    llvmir = _encode_string(llvmir)
    strbuf = c_char_p(llvmir)
    with ffi.OutputString() as errmsg:
        mod = ModuleRef(ffi.lib.LLVMPY_ParseAssembly(context, strbuf, errmsg), context)
        if errmsg:
            mod.close()
            raise RuntimeError('LLVM IR parsing error\n{0}'.format(errmsg))
    return mod