import math
from toolz import memoize
import numpy as np
from datashader.glyphs.glyph import Glyph
from datashader.resampling import infer_interval_breaks
from datashader.utils import isreal, ngjit, ngjit_parallel
import numba
from numba import cuda, prange
@ngjit
@self.expand_aggs_and_cols(append)
def extend_cpu(plot_height, plot_width, xs, ys, *aggs_and_cols):
    xverts = np.zeros(5, dtype=np.int32)
    yverts = np.zeros(5, dtype=np.int32)
    yincreasing = np.zeros(4, dtype=np.int8)
    eligible = np.ones(4, dtype=np.int8)
    intersect = np.zeros(4, dtype=np.int8)
    y_len, x_len = xs.shape
    for i in range(x_len - 1):
        for j in range(y_len - 1):
            perform_extend(i, j, plot_height, plot_width, xs, ys, xverts, yverts, yincreasing, eligible, intersect, *aggs_and_cols)