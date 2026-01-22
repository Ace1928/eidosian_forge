import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
def set_foreground(self, fg, isRGBA=None):
    super().set_foreground(fg, isRGBA)
    if len(self._rgb) == 3:
        self.ctx.set_source_rgb(*self._rgb)
    else:
        self.ctx.set_source_rgba(*self._rgb)