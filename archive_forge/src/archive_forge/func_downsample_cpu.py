import math
from toolz import memoize
import numpy as np
from datashader.glyphs.glyph import Glyph
from datashader.resampling import infer_interval_breaks
from datashader.utils import isreal, ngjit, ngjit_parallel
import numba
from numba import cuda, prange
@ngjit_parallel
@self.expand_aggs_and_cols(append)
def downsample_cpu(src_w, src_h, translate_x, translate_y, scale_x, scale_y, offset_x, offset_y, out_w, out_h, *aggs_and_cols):
    for out_j in prange(out_h):
        src_j0 = int(max(math.floor(scale_y * (out_j + 0.0) + translate_y - offset_y), 0))
        src_j1 = int(min(math.floor(scale_y * (out_j + 1.0) + translate_y - offset_y), src_h))
        for out_i in range(out_w):
            src_i0 = int(max(math.floor(scale_x * (out_i + 0.0) + translate_x - offset_x), 0))
            src_i1 = int(min(math.floor(scale_x * (out_i + 1.0) + translate_x - offset_x), src_w))
            for src_j in range(src_j0, src_j1):
                for src_i in range(src_i0, src_i1):
                    append(src_j, src_i, out_i, out_j, *aggs_and_cols)