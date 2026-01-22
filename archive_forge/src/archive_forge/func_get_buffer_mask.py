import re
import weakref
from ctypes import *
from io import open, BytesIO
import pyglet
from pyglet.gl import *
from pyglet.gl import gl_info
from pyglet.util import asbytes
from .codecs import ImageEncodeException, ImageDecodeException
from .codecs import registry as _codec_registry
from .codecs import add_default_codecs as _add_default_codecs
from .animation import Animation, AnimationFrame
from .buffer import *
from . import atlas
def get_buffer_mask(self):
    """Get a free bitmask buffer.

        A bitmask buffer is a buffer referencing a single bit in the stencil
        buffer.  If no bits are free, `ImageException` is raised.  Bits are
        released when the bitmask buffer is garbage collected.

        :rtype: :py:class:`~pyglet.image.BufferImageMask`
        """
    if self.free_stencil_bits is None:
        try:
            stencil_bits = GLint()
            glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, GL_STENCIL, GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE, stencil_bits)
            self.free_stencil_bits = list(range(stencil_bits.value))
        except GLException:
            pass
    if not self.free_stencil_bits:
        raise ImageException('No free stencil bits are available.')
    stencil_bit = self.free_stencil_bits.pop(0)
    x, y, width, height = self.get_viewport()
    bufimg = BufferImageMask(x, y, width, height)
    bufimg.stencil_bit = stencil_bit

    def release_buffer(ref, owner=self):
        owner.free_stencil_bits.insert(0, stencil_bit)
    self._refs.append(weakref.ref(bufimg, release_buffer))
    return bufimg