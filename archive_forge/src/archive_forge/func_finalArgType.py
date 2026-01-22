import ctypes
from OpenGL.platform import ctypesloader
from OpenGL._bytes import as_8_bit
import sys, logging
from OpenGL import _configflags
from OpenGL import logs, MODULE_ANNOTATIONS
def finalArgType(self, typ):
    """Retrieve a final type for arg-type"""
    if typ == ctypes.POINTER(None) and (not getattr(typ, 'final', False)):
        from OpenGL.arrays import ArrayDatatype
        return ArrayDatatype
    else:
        return typ