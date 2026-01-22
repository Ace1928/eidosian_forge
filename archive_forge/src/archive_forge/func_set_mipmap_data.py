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
def set_mipmap_data(self, level, data):
    """Set data for a mipmap level.

        Supplied data gives a compressed image for the given mipmap level.
        The image must be of the correct dimensions for the level
        (i.e., width >> level, height >> level); but this is not checked.  If
        any mipmap levels are specified, they are used; otherwise, mipmaps for
        `mipmapped_texture` are generated automatically.

        :Parameters:
            `level` : int
                Level of mipmap image to set.
            `data` : sequence
                String or array/list of bytes giving compressed image data.
                Data must be in same format as specified in constructor.

        """
    self.mipmap_data += [None] * (level - len(self.mipmap_data))
    self.mipmap_data[level - 1] = data