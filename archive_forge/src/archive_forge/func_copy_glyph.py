import copy
import os
import pathlib
import platform
from ctypes import *
from typing import List, Optional, Tuple
import math
import pyglet
from pyglet.font import base
from pyglet.image.codecs.wic import IWICBitmap, WICDecoder, GUID_WICPixelFormat32bppPBGRA
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
def copy_glyph(self, glyph, advance, offset):
    """This takes the existing glyph texture and puts it into a new Glyph with a new advance.
        Texture memory is shared between both glyphs."""
    new_glyph = base.Glyph(glyph.x, glyph.y, glyph.z, glyph.width, glyph.height, glyph.owner)
    new_glyph.set_bearings(glyph.baseline, glyph.lsb, advance, offset.advanceOffset, offset.ascenderOffset)
    return new_glyph