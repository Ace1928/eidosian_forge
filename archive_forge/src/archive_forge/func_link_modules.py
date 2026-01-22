from ctypes import c_int, c_char_p, POINTER
from llvmlite.binding import ffi
def link_modules(dst, src):
    with ffi.OutputString() as outerr:
        err = ffi.lib.LLVMPY_LinkModules(dst, src, outerr)
        src.detach()
        if err:
            raise RuntimeError(str(outerr))