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
def print_figure_impl(fh):
    if is_eps:
        print('%!PS-Adobe-3.0 EPSF-3.0', file=fh)
    else:
        print('%!PS-Adobe-3.0', file=fh)
        if papertype != 'figure':
            print(f'%%DocumentPaperSizes: {papertype}', file=fh)
        print('%%Pages: 1', file=fh)
    print(f'%%LanguageLevel: 3\n{dsc_comments}\n%%Orientation: {orientation.name}\n{get_bbox_header(bbox)[0]}\n%%EndComments\n', end='', file=fh)
    Ndict = len(_psDefs)
    print('%%BeginProlog', file=fh)
    if not mpl.rcParams['ps.useafm']:
        Ndict += len(ps_renderer._character_tracker.used)
    print('/mpldict %d dict def' % Ndict, file=fh)
    print('mpldict begin', file=fh)
    print('\n'.join(_psDefs), file=fh)
    if not mpl.rcParams['ps.useafm']:
        for font_path, chars in ps_renderer._character_tracker.used.items():
            if not chars:
                continue
            fonttype = mpl.rcParams['ps.fonttype']
            if len(chars) > 255:
                fonttype = 42
            fh.flush()
            if fonttype == 3:
                fh.write(_font_to_ps_type3(font_path, chars))
            else:
                _font_to_ps_type42(font_path, chars, fh)
    print('end', file=fh)
    print('%%EndProlog', file=fh)
    if not is_eps:
        print('%%Page: 1 1', file=fh)
    print('mpldict begin', file=fh)
    print('%s translate' % _nums_to_str(xo, yo), file=fh)
    if rotation:
        print('%d rotate' % rotation, file=fh)
    print(f'0 0 {_nums_to_str(width * 72, height * 72)} rectclip', file=fh)
    print(self._pswriter.getvalue(), file=fh)
    print('end', file=fh)
    print('showpage', file=fh)
    if not is_eps:
        print('%%EOF', file=fh)
    fh.flush()