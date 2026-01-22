import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
def get_font_metrics(self, size, dpi):
    if self.set_char_size(size, dpi):
        metrics = self.ft_face.contents.size.contents.metrics
        if metrics.ascender == 0 and metrics.descender == 0:
            return self._get_font_metrics_workaround()
        else:
            return FreeTypeFontMetrics(ascent=int(f26p6_to_float(metrics.ascender)), descent=int(f26p6_to_float(metrics.descender)))
    else:
        return self._get_font_metrics_workaround()