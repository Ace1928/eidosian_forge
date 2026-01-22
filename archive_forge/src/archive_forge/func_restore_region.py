import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
def restore_region(self, region):
    surface = self._renderer.gc.ctx.get_target()
    if not isinstance(surface, cairo.ImageSurface):
        raise RuntimeError('restore_region only works when rendering to an ImageSurface')
    surface.flush()
    sw = surface.get_width()
    sh = surface.get_height()
    sly, slx = region._slices
    np.frombuffer(surface.get_data(), np.uint32).reshape((sh, sw))[sly, slx] = region._data
    surface.mark_dirty_rectangle(slx.start, sly.start, slx.stop - slx.start, sly.stop - sly.start)