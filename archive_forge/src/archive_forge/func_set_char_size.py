import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
def set_char_size(self, size, dpi):
    face_size = float_to_f26p6(size)
    try:
        FT_Set_Char_Size(self.ft_face, 0, face_size, dpi, dpi)
        return True
    except FreeTypeError as e:
        if e.errcode == 23:
            return False
        else:
            raise