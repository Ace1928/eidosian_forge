import ctypes
import weakref
from OpenGL._bytes import long, integer_types
def ptrAsArray(self, ptr, length, type):
    """Copy length values from ptr into new array of given type"""
    result = type.zeros((length,))
    for i in range(length):
        result[i] = ptr[i]
    return result