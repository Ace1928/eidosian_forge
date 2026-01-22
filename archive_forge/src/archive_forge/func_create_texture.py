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
def create_texture(self, cls, rectangle=False):
    """Create a texture containing this image.

        :Parameters:
            `cls` : class (subclass of Texture)
                Class to construct.
            `rectangle` : bool
                Unused. kept for compatibility.

                .. versionadded:: 1.1

        :rtype: cls or cls.region_class
        """
    internalformat = self._get_internalformat(self._desired_format)
    texture = cls.create(self.width, self.height, GL_TEXTURE_2D, internalformat, blank_data=False)
    if self.anchor_x or self.anchor_y:
        texture.anchor_x = self.anchor_x
        texture.anchor_y = self.anchor_y
    self.blit_to_texture(texture.target, texture.level, self.anchor_x, self.anchor_y, 0, None)
    return texture