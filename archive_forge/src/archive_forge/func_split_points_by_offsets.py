from __future__ import annotations
from itertools import chain
from typing import TYPE_CHECKING
import numpy as np
from contourpy.typecheck import check_code_array, check_offset_array, check_point_array
from contourpy.types import CLOSEPOLY, LINETO, MOVETO, code_dtype, offset_dtype, point_dtype
def split_points_by_offsets(points: cpy.PointArray, offsets: cpy.OffsetArray) -> list[cpy.PointArray]:
    """Split a point array at locations specified by an offset array into a list of point arrays.
    """
    check_point_array(points)
    check_offset_array(offsets)
    if len(offsets) > 2:
        return np.split(points, offsets[1:-1])
    else:
        return [points]