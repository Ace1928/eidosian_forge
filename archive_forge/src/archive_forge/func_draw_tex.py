import codecs
import datetime
from enum import Enum
import functools
from io import StringIO
import itertools
import logging
import os
import pathlib
import shutil
from tempfile import TemporaryDirectory
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _path, _text_helpers
from matplotlib._afm import AFM
from matplotlib.backend_bases import (
from matplotlib.cbook import is_writable_file_like, file_requires_unicode
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LOAD_NO_SCALE, FT2Font
from matplotlib._ttconv import convert_ttf_to_ps
from matplotlib._mathtext_data import uni2type1
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_mixed import MixedModeRenderer
from . import _backend_pdf_ps
@_log_if_debug_on
def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
    if self._is_transparent(gc.get_rgb()):
        return
    if not hasattr(self, 'psfrag'):
        self._logwarn_once("The PS backend determines usetex status solely based on rcParams['text.usetex'] and does not support having usetex=True only for some elements; this element will thus be rendered as if usetex=False.")
        self.draw_text(gc, x, y, s, prop, angle, False, mtext)
        return
    w, h, bl = self.get_text_width_height_descent(s, prop, ismath='TeX')
    fontsize = prop.get_size_in_points()
    thetext = 'psmarker%d' % self.textcnt
    color = _nums_to_str(*gc.get_rgb()[:3], sep=',')
    fontcmd = {'sans-serif': '{\\sffamily %s}', 'monospace': '{\\ttfamily %s}'}.get(mpl.rcParams['font.family'][0], '{\\rmfamily %s}')
    s = fontcmd % s
    tex = '\\color[rgb]{%s} %s' % (color, s)
    rangle = np.radians(angle + 90)
    pos = _nums_to_str(x - bl * np.cos(rangle), y - bl * np.sin(rangle))
    self.psfrag.append('\\psfrag{%s}[bl][bl][1][%f]{\\fontsize{%f}{%f}%s}' % (thetext, angle, fontsize, fontsize * 1.25, tex))
    self._pswriter.write(f'gsave\n{pos} moveto\n({thetext})\nshow\ngrestore\n')
    self.textcnt += 1