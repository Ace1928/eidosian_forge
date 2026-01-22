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
def get_glyph_metrics(self, font_face, indices, count):
    """Returns a list of tuples with the following metrics per indice:
            (glyph width, glyph height, lsb, advanceWidth)
        """
    glyph_metrics = (DWRITE_GLYPH_METRICS * count)()
    font_face.GetDesignGlyphMetrics(indices, count, glyph_metrics, False)
    metrics_out = []
    for metric in glyph_metrics:
        glyph_width = metric.advanceWidth - metric.leftSideBearing - metric.rightSideBearing
        if glyph_width == 0:
            glyph_width = 1
        glyph_height = metric.advanceHeight - metric.topSideBearing - metric.bottomSideBearing
        lsb = metric.leftSideBearing
        bsb = metric.bottomSideBearing
        advance_width = metric.advanceWidth
        metrics_out.append((glyph_width, glyph_height, lsb, advance_width, bsb))
    return metrics_out