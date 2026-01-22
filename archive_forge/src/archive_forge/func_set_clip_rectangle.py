import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
def set_clip_rectangle(self, rectangle):
    if not rectangle:
        return
    x, y, w, h = np.round(rectangle.bounds)
    ctx = self.ctx
    ctx.new_path()
    ctx.rectangle(x, self.renderer.height - h - y, w, h)
    ctx.clip()