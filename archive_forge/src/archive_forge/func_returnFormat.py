from OpenGL.raw.GL.VERSION import GL_1_1 as _simple
from OpenGL import arrays
from OpenGL import error
from OpenGL import _configflags
import ctypes
def returnFormat(data, type):
    """Perform compatibility conversion for PyOpenGL 2.x image-as string results
    
    Uses OpenGL.UNSIGNED_BYTE_IMAGES_AS_STRING to control whether to perform the 
    conversions.
    """
    if _configflags.UNSIGNED_BYTE_IMAGES_AS_STRING:
        if type == _simple.GL_UNSIGNED_BYTE:
            if hasattr(data, 'tobytes'):
                return data.tobytes()
            elif hasattr(data, 'tostring'):
                return data.tostring()
            elif hasattr(data, 'raw'):
                return data.raw
            elif hasattr(data, '_type_'):
                s = ctypes.string_at(ctypes.cast(data, ctypes.c_voidp), ctypes.sizeof(data))
                result = s[:]
                return result
    return data