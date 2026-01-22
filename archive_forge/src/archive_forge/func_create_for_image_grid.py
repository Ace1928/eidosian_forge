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
@classmethod
def create_for_image_grid(cls, grid, internalformat=GL_RGBA):
    texture_array = cls.create(grid[0].width, grid[0].height, internalformat, max_depth=len(grid))
    texture_array.allocate(*grid[:])
    return texture_array