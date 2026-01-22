from OpenGL.raw.GL import _types
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL.arrays import formathandler
import ctypes
from OpenGL._bytes import long, integer_types
def registerEquivalent(self, typ, base):
    """Register a sub-class for handling as the base-type"""
    global TARGET_TYPE_TUPLE
    for source in (DEFAULT_TYPES, TARGET_TYPES, BYTE_SIZES):
        if base in source:
            source[typ] = source[base]
    if base in TARGET_TYPES:
        TARGET_TYPE_TUPLE = TARGET_TYPE_TUPLE + (base,)