from OpenGL.raw import GLU as _simple
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL.platform import createBaseFunction
from OpenGL.GLU import glustruct
from OpenGL import arrays, wrapper
from OpenGL.platform import PLATFORM
from OpenGL.lazywrapper import lazy as _lazy
import ctypes
@_lazy(createBaseFunction('gluNewTess', dll=GLU, resultType=ctypes.POINTER(GLUtesselator), doc='gluNewTess(  ) -> POINTER(GLUtesselator)'))
def gluNewTess(baseFunction):
    """Get a new tessellator object (just unpacks the pointer for you)"""
    return baseFunction()[0]