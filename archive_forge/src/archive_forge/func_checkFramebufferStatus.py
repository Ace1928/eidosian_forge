from OpenGL.extensions import alternate
from OpenGL.GL.ARB.framebuffer_object import *
from OpenGL.GL.EXT.framebuffer_object import *
from OpenGL.GL.EXT.framebuffer_multisample import *
from OpenGL.GL.EXT.framebuffer_blit import *
def checkFramebufferStatus():
    """Utility method to check status and raise errors"""
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status == GL_FRAMEBUFFER_COMPLETE:
        return True
    from OpenGL.error import GLError
    description = None
    for error_constant in [GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT, GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT, GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS, GL_FRAMEBUFFER_INCOMPLETE_FORMATS, GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER, GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER, GL_FRAMEBUFFER_UNSUPPORTED]:
        if status == error_constant:
            status = error_constant
            description = str(status)
    raise GLError(err=status, result=status, baseOperation=glCheckFramebufferStatus, description=description)