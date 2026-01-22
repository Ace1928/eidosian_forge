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
@_api.deprecated('3.8', alternative='buffer_rgba')
def tostring_rgb(self):
    """
        Get the image as RGB `bytes`.

        `draw` must be called at least once before this function will work and
        to update the renderer for any subsequent changes to the Figure.
        """
    return self.renderer.tostring_rgb()