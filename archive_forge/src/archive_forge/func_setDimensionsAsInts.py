from OpenGL.raw.GL.VERSION import GL_1_1,GL_1_2, GL_3_0
from OpenGL import images, arrays, wrapper
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,integer_types
from OpenGL.raw.GL import _types
import ctypes
def setDimensionsAsInts(baseOperation):
    """Set arguments with names in INT_DIMENSION_NAMES to asInt processing"""
    baseOperation = asWrapper(baseOperation)
    argNames = getattr(baseOperation, 'pyConverterNames', baseOperation.argNames)
    for i, argName in enumerate(argNames):
        if argName in INT_DIMENSION_NAMES:
            baseOperation.setPyConverter(argName, asIntConverter)
    return baseOperation