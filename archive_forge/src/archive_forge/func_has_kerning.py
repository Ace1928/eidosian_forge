from ctypes import *
from .base import FontException
import pyglet.lib
def has_kerning(self):
    return self.face_flags & FT_FACE_FLAG_KERNING