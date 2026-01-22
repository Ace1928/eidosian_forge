import pyglet
from pyglet.gl import *
def attach_renderbuffer(self, renderbuffer, target=GL_FRAMEBUFFER, attachment=GL_COLOR_ATTACHMENT0):
    """Attach a Renderbuffer to the Framebuffer

        :Parameters:
            `renderbuffer` : pyglet.image.Renderbuffer
                Specifies the Renderbuffer to attach to the framebuffer attachment
                point named by attachment.
            `target` : int
                Specifies the framebuffer target. target must be GL_DRAW_FRAMEBUFFER,
                GL_READ_FRAMEBUFFER, or GL_FRAMEBUFFER. GL_FRAMEBUFFER is equivalent
                to GL_DRAW_FRAMEBUFFER.
            `attachment` : int
                Specifies the attachment point of the framebuffer. attachment must be
                GL_COLOR_ATTACHMENTi, GL_DEPTH_ATTACHMENT, GL_STENCIL_ATTACHMENT or
                GL_DEPTH_STENCIL_ATTACHMENT.
        """
    self.bind()
    glFramebufferRenderbuffer(target, attachment, GL_RENDERBUFFER, renderbuffer.id)
    self._attachment_types |= attachment
    self._width = max(renderbuffer.width, self._width)
    self._height = max(renderbuffer.height, self._height)
    self.unbind()