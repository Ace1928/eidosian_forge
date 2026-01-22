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
def map_onto_pixel_snap(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x, y):
    """Map points onto pixel grid.

        Points falling on upper bound are mapped into previous bin.

        If the line has been clipped, x and y will have been
        computed to lie on the bounds; we compare point and bounds
        in integer space to avoid fp error. In contrast, with
        auto-ranging, a point on the bounds will be the same
        floating point number as the bound, so comparison in fp
        representation of continuous space or in integer space
        doesn't change anything.
        """
    xx = int(x_mapper(x) * sx + tx)
    yy = int(y_mapper(y) * sy + ty)
    xxmax = round(x_mapper(xmax) * sx + tx)
    yymax = round(y_mapper(ymax) * sy + ty)
    return (xx - 1 if xx == xxmax else xx, yy - 1 if yy == yymax else yy)