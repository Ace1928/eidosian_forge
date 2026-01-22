import itertools
import numpy as np
import pandas as pd
import param
from ..core import Dataset
from ..core.boundingregion import BoundingBox
from ..core.data import PandasInterface, default_datatype
from ..core.operation import Operation
from ..core.sheetcoords import Slice
from ..core.util import (
def quadratic_bezier(start, end, c0=(0, 0), c1=(0, 0), steps=50):
    """
    Compute quadratic bezier spline given start and end coordinate and
    two control points.
    """
    steps = np.linspace(0, 1, steps)
    sx, sy = start
    ex, ey = end
    cx0, cy0 = c0
    cx1, cy1 = c1
    xs = (1 - steps) ** 3 * sx + 3 * (1 - steps) ** 2 * steps * cx0 + 3 * (1 - steps) * steps ** 2 * cx1 + steps ** 3 * ex
    ys = (1 - steps) ** 3 * sy + 3 * (1 - steps) ** 2 * steps * cy0 + 3 * (1 - steps) * steps ** 2 * cy1 + steps ** 3 * ey
    return np.column_stack([xs, ys])