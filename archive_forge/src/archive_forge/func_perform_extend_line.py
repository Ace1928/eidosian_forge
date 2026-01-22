from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
@ngjit
@expand_aggs_and_cols
def perform_extend_line(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, buffer, *aggs_and_cols):
    x0 = xs[i, j]
    y0 = ys[j]
    x1 = xs[i, j + 1]
    y1 = ys[j + 1]
    segment_start = j == 0 or isnull(xs[i, j - 1]) or isnull(ys[j - 1])
    segment_end = j == len(ys) - 2 or isnull(xs[i, j + 2]) or isnull(ys[j + 2])
    if segment_start or use_2_stage_agg:
        xm = 0.0
        ym = 0.0
    else:
        xm = xs[i, j - 1]
        ym = ys[j - 1]
    draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start, segment_end, x0, x1, y0, y1, xm, ym, buffer, *aggs_and_cols)