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
def get_bbox_header(lbrt, rotated=False):
    """
    Return a postscript header string for the given bbox lbrt=(l, b, r, t).
    Optionally, return rotate command.
    """
    l, b, r, t = lbrt
    if rotated:
        rotate = f'{l + r:.2f} {0:.2f} translate\n90 rotate'
    else:
        rotate = ''
    bbox_info = '%%%%BoundingBox: %d %d %d %d' % (l, b, np.ceil(r), np.ceil(t))
    hires_bbox_info = f'%%HiResBoundingBox: {l:.6f} {b:.6f} {r:.6f} {t:.6f}'
    return ('\n'.join([bbox_info, hires_bbox_info]), rotate)