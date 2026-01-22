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
def get_glyphs_with_font(self, font, s, glyph_map=None, return_new_glyphs_only=False):
    """
        Convert string *s* to vertices and codes using the provided ttf font.
        """
    if glyph_map is None:
        glyph_map = OrderedDict()
    if return_new_glyphs_only:
        glyph_map_new = OrderedDict()
    else:
        glyph_map_new = glyph_map
    xpositions = []
    glyph_ids = []
    for item in _text_helpers.layout(s, font):
        char_id = self._get_char_id(item.ft_object, ord(item.char))
        glyph_ids.append(char_id)
        xpositions.append(item.x)
        if char_id not in glyph_map:
            glyph_map_new[char_id] = item.ft_object.get_path()
    ypositions = [0] * len(xpositions)
    sizes = [1.0] * len(xpositions)
    rects = []
    return (list(zip(glyph_ids, xpositions, ypositions, sizes)), glyph_map_new, rects)