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
def start_filter(self):
    """
        Start filtering. It simply creates a new canvas (the old one is saved).
        """
    self._filter_renderers.append(self._renderer)
    self._renderer = _RendererAgg(int(self.width), int(self.height), self.dpi)
    self._update_methods()