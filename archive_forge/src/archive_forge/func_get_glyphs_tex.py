from collections import OrderedDict
import logging
import urllib.parse
import numpy as np
from matplotlib import _text_helpers, dviread
from matplotlib.font_manager import (
from matplotlib.ft2font import LOAD_NO_HINTING, LOAD_TARGET_LIGHT
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
def get_glyphs_tex(self, prop, s, glyph_map=None, return_new_glyphs_only=False):
    """Convert the string *s* to vertices and codes using usetex mode."""
    dvifile = TexManager().make_dvi(s, self.FONT_SCALE)
    with dviread.Dvi(dvifile, self.DPI) as dvi:
        page, = dvi
    if glyph_map is None:
        glyph_map = OrderedDict()
    if return_new_glyphs_only:
        glyph_map_new = OrderedDict()
    else:
        glyph_map_new = glyph_map
    glyph_ids, xpositions, ypositions, sizes = ([], [], [], [])
    for text in page.text:
        font = get_font(text.font_path)
        char_id = self._get_char_id(font, text.glyph)
        if char_id not in glyph_map:
            font.clear()
            font.set_size(self.FONT_SCALE, self.DPI)
            glyph_name_or_index = text.glyph_name_or_index
            if isinstance(glyph_name_or_index, str):
                index = font.get_name_index(glyph_name_or_index)
                font.load_glyph(index, flags=LOAD_TARGET_LIGHT)
            elif isinstance(glyph_name_or_index, int):
                self._select_native_charmap(font)
                font.load_char(glyph_name_or_index, flags=LOAD_TARGET_LIGHT)
            else:
                raise TypeError(f'Glyph spec of unexpected type: {glyph_name_or_index!r}')
            glyph_map_new[char_id] = font.get_path()
        glyph_ids.append(char_id)
        xpositions.append(text.x)
        ypositions.append(text.y)
        sizes.append(text.font_size / self.FONT_SCALE)
    myrects = []
    for ox, oy, h, w in page.boxes:
        vert1 = [(ox, oy), (ox + w, oy), (ox + w, oy + h), (ox, oy + h), (ox, oy), (0, 0)]
        code1 = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        myrects.append((vert1, code1))
    return (list(zip(glyph_ids, xpositions, ypositions, sizes)), glyph_map_new, myrects)