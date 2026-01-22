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
def render_using_layout(self, text):
    """This will render text given the built in DirectWrite layout. This process allows us to take
        advantage of color glyphs and fallback handling that is built into DirectWrite.
        This can also handle shaping and many other features if you want to render directly to a texture."""
    text_layout = self.font.create_text_layout(text)
    layout_metrics = DWRITE_TEXT_METRICS()
    text_layout.GetMetrics(byref(layout_metrics))
    width = int(math.ceil(layout_metrics.width))
    height = int(math.ceil(layout_metrics.height))
    if width == 0 or height == 0:
        return None
    self._create_bitmap(width, height)
    point = D2D_POINT_2F(0, 0)
    self._render_target.BeginDraw()
    self._render_target.Clear(transparent)
    self._render_target.DrawTextLayout(point, text_layout, self._brush, self.draw_options)
    self._render_target.EndDraw(None, None)
    image = wic_decoder.get_image(self._bitmap)
    glyph = self.font.create_glyph(image)
    glyph.set_bearings(-self.font.descent, 0, int(math.ceil(layout_metrics.width)))
    return glyph