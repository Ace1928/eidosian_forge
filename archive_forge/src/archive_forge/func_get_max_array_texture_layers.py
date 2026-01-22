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
def get_max_array_texture_layers():
    """Query the maximum TextureArray depth"""
    max_layers = c_int()
    glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, max_layers)
    return max_layers.value