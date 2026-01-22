import pyglet
from pyglet.gl import *
def attach_texture(self, texture, target=GL_FRAMEBUFFER, attachment=GL_COLOR_ATTACHMENT0):
    """Attach a Texture to the Framebuffer

        :Parameters:
            `texture` : pyglet.image.Texture
                Specifies the texture object to attach to the framebuffer attachment
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
    glFramebufferTexture(target, attachment, texture.id, texture.level)
    self._attachment_types |= attachment
    self._width = max(texture.width, self._width)
    self._height = max(texture.height, self._height)
    self.unbind()