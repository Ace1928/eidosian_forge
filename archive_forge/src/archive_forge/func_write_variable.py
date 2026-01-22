import ctypes, ctypes.util, operator, sys
from . import model
def write_variable(self, BType, name, value):
    new_ctypes_obj = BType._to_ctypes(value)
    ctypes_obj = BType._ctype.in_dll(self.cdll, name)
    ctypes.memmove(ctypes.addressof(ctypes_obj), ctypes.addressof(new_ctypes_obj), ctypes.sizeof(BType._ctype))