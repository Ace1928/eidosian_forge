from OpenGL.arrays.arraydatatype import ArrayDatatype
from OpenGL.arrays.formathandler import FormatHandler
from OpenGL.raw.GL import _types 
from OpenGL import error
from OpenGL._bytes import bytes,unicode,as_8_bit
import ctypes,logging
from OpenGL._bytes import long, integer_types
import weakref
from OpenGL import acceleratesupport
def mapVBO(vbo, access=35002):
    """Map the given buffer into a numpy array...

    Method taken from:
     http://www.mail-archive.com/numpy-discussion@lists.sourceforge.net/msg01161.html

    This should be considered an *experimental* API,
    it is not guaranteed to be available in future revisions
    of this library!
    
    Simplification to use ctypes cast from comment by 'sashimi' on my blog...
    """
    from numpy import frombuffer
    vp = vbo.implementation.glMapBuffer(vbo.target, access)
    vp_array = ctypes.cast(vp, ctypes.POINTER(ctypes.c_byte * vbo.size))
    array = frombuffer(vp_array, 'B')
    _cleaners[vbo] = weakref.ref(array, _cleaner(vbo))
    return array