from OpenGL.raw.GL.VERSION import GL_1_1 as _simple
from OpenGL import arrays
from OpenGL import error
from OpenGL import _configflags
import ctypes
def createTargetArray(format, dims, type):
    """Create storage array for given parameters
    
    If storage type requires > 1 unit per format pixel, then dims will be
    extended by 1, so in the common case of RGB and GL_UNSIGNED_BYTE you 
    will wind up with an array of dims + (3,) dimensions.  See
    COMPONENT_COUNTS for table which controls which formats produce
    larger dimensions.  The secondary table TIGHT_PACK_FORMATS overrides 
    this case, so that image formats registered as TIGHT_PACK_FORMATS
    only ever return a dims-shaped value.  TIGHT_PACK_FORMATS will raise 
    ValueErrors if they are used with a format that does not have the same 
    number of components as they define.
    
    Note that the base storage type must provide a zeros method.  The zeros
    method relies on their being a registered default array-implementation for 
    the storage type.  The default installation of OpenGL-ctypes will use 
    Numpy arrays for returning the result.
    """
    componentCount = formatToComponentCount(format)
    if componentCount > 1:
        if type not in TIGHT_PACK_FORMATS:
            dims += (componentCount,)
        elif TIGHT_PACK_FORMATS[type] < componentCount:
            raise ValueError('Image type: %s supports %s components, but format %s requires %s components' % (type, TIGHT_PACK_FORMATS[type], format, componentCount))
    arrayType = arrays.GL_CONSTANT_TO_ARRAY_TYPE[TYPE_TO_ARRAYTYPE.get(type, type)]
    return arrayType.zeros(dims)