import math
from toolz import memoize
import numpy as np
from datashader.glyphs.glyph import Glyph
from datashader.resampling import infer_interval_breaks
from datashader.utils import isreal, ngjit, ngjit_parallel
import numba
from numba import cuda, prange
def is_upsample(self, source, x, y, name, x_range, y_range, out_w, out_h):
    src_w = len(source[x])
    if x_range is None:
        upsample_width = out_w >= src_w
    else:
        out_x0, out_x1 = x_range
        src_x0, src_x1 = self._compute_bounds_from_1d_centers(source, x, maybe_expand=False, orient=False)
        src_xbinsize = math.fabs((src_x1 - src_x0) / src_w)
        out_xbinsize = math.fabs((out_x1 - out_x0) / out_w)
        upsample_width = src_xbinsize >= out_xbinsize
    src_h = len(source[y])
    if y_range is None:
        upsample_height = out_h >= src_h
    else:
        out_y0, out_y1 = y_range
        src_y0, src_y1 = self._compute_bounds_from_1d_centers(source, y, maybe_expand=False, orient=False)
        src_ybinsize = math.fabs((src_y1 - src_y0) / src_h)
        out_ybinsize = math.fabs((out_y1 - out_y0) / out_h)
        upsample_height = src_ybinsize >= out_ybinsize
    return (upsample_width, upsample_height)