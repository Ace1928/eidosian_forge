import math
from toolz import memoize
import numpy as np
from datashader.glyphs.glyph import Glyph
from datashader.resampling import infer_interval_breaks
from datashader.utils import isreal, ngjit, ngjit_parallel
import numba
from numba import cuda, prange
@ngjit
def build_scale_translate(out_size, out0, out1, src_size, src0, src1):
    translate_y = src_size * (out0 - src0) / (src1 - src0)
    scale_y = src_size * (out1 - out0) / (out_size * (src1 - src0))
    return (scale_y, translate_y)