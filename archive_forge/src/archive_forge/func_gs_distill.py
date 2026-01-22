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
def gs_distill(tmpfile, eps=False, ptype='letter', bbox=None, rotated=False):
    """
    Use ghostscript's pswrite or epswrite device to distill a file.
    This yields smaller files without illegal encapsulated postscript
    operators. The output is low-level, converting text to outlines.
    """
    if eps:
        paper_option = ['-dEPSCrop']
    elif ptype == 'figure':
        paper_option = [f'-dDEVICEWIDTHPOINTS={bbox[2]}', f'-dDEVICEHEIGHTPOINTS={bbox[3]}']
    else:
        paper_option = [f'-sPAPERSIZE={ptype}']
    psfile = tmpfile + '.ps'
    dpi = mpl.rcParams['ps.distiller.res']
    cbook._check_and_log_subprocess([mpl._get_executable_info('gs').executable, '-dBATCH', '-dNOPAUSE', '-r%d' % dpi, '-sDEVICE=ps2write', *paper_option, f'-sOutputFile={psfile}', tmpfile], _log)
    os.remove(tmpfile)
    shutil.move(psfile, tmpfile)
    if eps:
        pstoeps(tmpfile, bbox, rotated=rotated)