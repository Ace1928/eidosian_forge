import logging
from OpenGL.GLES2 import *
from OpenGL._bytes import bytes,unicode,as_8_bit
def check_linked(self):
    """Check link status for this program
        
        raises RuntimeError on failures
        """
    link_status = glGetProgramiv(self, GL_LINK_STATUS)
    if link_status == GL_FALSE:
        raise RuntimeError('Link failure (%s): %s' % (link_status, glGetProgramInfoLog(self)))
    return self