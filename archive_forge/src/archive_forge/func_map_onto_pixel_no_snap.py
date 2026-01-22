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
def map_onto_pixel_no_snap(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x, y):
    xx = x_mapper(x) * sx + tx - 0.5
    yy = y_mapper(y) * sy + ty - 0.5
    return (xx, yy)