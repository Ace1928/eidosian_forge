import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
@classmethod
def from_fontconfig(cls, match):
    if match.face is not None:
        FT_Reference_Face(match.face)
        return cls(match.face)
    else:
        if not match.file:
            raise base.FontException(f'No filename for "{match.name}"')
        return cls.from_file(match.file)