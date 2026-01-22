import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
def set_clip_path(self, path):
    if not path:
        return
    tpath, affine = path.get_transformed_path_and_affine()
    ctx = self.ctx
    ctx.new_path()
    affine = affine + Affine2D().scale(1, -1).translate(0, self.renderer.height)
    _append_path(ctx, tpath, affine)
    ctx.clip()