import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_uint8, c_uint64, c_size_t
from llvmlite.binding import ffi, targets
def export_symbol(self, name):
    """
        During linking, extract the address of a symbol that was defined in one
        of the compilation units.

        This allows getting symbols, functions or global variables, out of the
        JIT linked library. The addresses will be
        available when the link method is called.
        """
    self.__exports.add(str(name))
    return self