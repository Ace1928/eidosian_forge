from __future__ import annotations
from itertools import chain
from typing import TYPE_CHECKING
import numpy as np
from contourpy.typecheck import check_code_array, check_offset_array, check_point_array
from contourpy.types import CLOSEPOLY, LINETO, MOVETO, code_dtype, offset_dtype, point_dtype
def split_points_at_nan(points: cpy.PointArray) -> list[cpy.PointArray]:
    """Split a points array at NaNs into a list of point arrays.
    """
    check_point_array(points)
    nan_offsets = np.nonzero(np.isnan(points[:, 0]))[0]
    if len(nan_offsets) == 0:
        return [points]
    else:
        nan_offsets = np.concatenate(([-1], nan_offsets, [len(points)]))
        return [points[s + 1:e] for s, e in zip(nan_offsets[:-1], nan_offsets[1:])]