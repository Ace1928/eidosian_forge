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
def create_hatch(self, hatch):
    sidelen = 72
    if hatch in self._hatches:
        return self._hatches[hatch]
    name = 'H%d' % len(self._hatches)
    linewidth = mpl.rcParams['hatch.linewidth']
    pageheight = self.height * 72
    self._pswriter.write(f'  << /PatternType 1\n     /PaintType 2\n     /TilingType 2\n     /BBox[0 0 {sidelen:d} {sidelen:d}]\n     /XStep {sidelen:d}\n     /YStep {sidelen:d}\n\n     /PaintProc {{\n        pop\n        {linewidth:g} setlinewidth\n{self._convert_path(Path.hatch(hatch), Affine2D().scale(sidelen), simplify=False)}\n        gsave\n        fill\n        grestore\n        stroke\n     }} bind\n   >>\n   matrix\n   0 {pageheight:g} translate\n   makepattern\n   /{name} exch def\n')
    self._hatches[hatch] = name
    return name