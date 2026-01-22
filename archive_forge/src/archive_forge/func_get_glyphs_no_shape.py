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
def get_glyphs_no_shape(self, text):
    """This differs in that it does not attempt to shape the text at all. May be useful in cases where your font
        has no special shaping requirements, spacing is the same, or some other reason where faster performance is
        wanted and you can get away with this."""
    if not self._glyph_renderer:
        self._glyph_renderer = self.glyph_renderer_class(self)
        self._empty_glyph = self._glyph_renderer.render_using_layout(' ')
    glyphs = []
    for c in text:
        if c == '\t':
            c = ' '
        if c not in self.glyphs:
            self.glyphs[c] = self._glyph_renderer.render_using_layout(c)
            if not self.glyphs[c]:
                self.glyphs[c] = self._empty_glyph
        glyphs.append(self.glyphs[c])
    return glyphs