from contextlib import nullcontext
from math import radians, cos, sin
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.backend_bases import (
from matplotlib.font_manager import fontManager as _fontManager, get_font
from matplotlib.ft2font import (LOAD_FORCE_AUTOHINT, LOAD_NO_HINTING,
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxBase
from matplotlib.backends._backend_agg import RendererAgg as _RendererAgg
def get_hinting_flag():
    mapping = {'default': LOAD_DEFAULT, 'no_autohint': LOAD_NO_AUTOHINT, 'force_autohint': LOAD_FORCE_AUTOHINT, 'no_hinting': LOAD_NO_HINTING, True: LOAD_FORCE_AUTOHINT, False: LOAD_NO_HINTING, 'either': LOAD_DEFAULT, 'native': LOAD_NO_AUTOHINT, 'auto': LOAD_FORCE_AUTOHINT, 'none': LOAD_NO_HINTING}
    return mapping[mpl.rcParams['text.hinting']]