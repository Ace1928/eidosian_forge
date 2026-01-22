from __future__ import annotations
import abc
import copy
import enum
import functools
import logging
import os
import re
import types
import unicodedata
import string
import typing as T
from typing import NamedTuple
import numpy as np
from pyparsing import (
import matplotlib as mpl
from . import cbook
from ._mathtext_data import (
from .font_manager import FontProperties, findfont, get_font
from .ft2font import FT2Font, FT2Image, KERNING_DEFAULT
from packaging.version import parse as parse_version
from pyparsing import __version__ as pyparsing_version
def to_raster(self, *, antialiased: bool) -> RasterParse:
    xmin = min([*[ox + info.metrics.xmin for ox, oy, info in self.glyphs], *[x1 for x1, y1, x2, y2 in self.rects], 0]) - 1
    ymin = min([*[oy - info.metrics.ymax for ox, oy, info in self.glyphs], *[y1 for x1, y1, x2, y2 in self.rects], 0]) - 1
    xmax = max([*[ox + info.metrics.xmax for ox, oy, info in self.glyphs], *[x2 for x1, y1, x2, y2 in self.rects], 0]) + 1
    ymax = max([*[oy - info.metrics.ymin for ox, oy, info in self.glyphs], *[y2 for x1, y1, x2, y2 in self.rects], 0]) + 1
    w = xmax - xmin
    h = ymax - ymin - self.box.depth
    d = ymax - ymin - self.box.height
    image = FT2Image(np.ceil(w), np.ceil(h + max(d, 0)))
    shifted = ship(self.box, (-xmin, -ymin))
    for ox, oy, info in shifted.glyphs:
        info.font.draw_glyph_to_bitmap(image, ox, oy - info.metrics.iceberg, info.glyph, antialiased=antialiased)
    for x1, y1, x2, y2 in shifted.rects:
        height = max(int(y2 - y1) - 1, 0)
        if height == 0:
            center = (y2 + y1) / 2
            y = int(center - (height + 1) / 2)
        else:
            y = int(y1)
        image.draw_rect_filled(int(x1), y, np.ceil(x2), y + height)
    return RasterParse(0, 0, w, h + d, d, image)