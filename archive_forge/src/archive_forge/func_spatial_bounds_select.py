import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
def spatial_bounds_select(xvals, yvals, bounds):
    x0, y0, x1, y1 = bounds
    return np.array([(x0 <= np.nanmin(xs)) & (y0 <= np.nanmin(ys)) & (x1 >= np.nanmax(xs)) & (y1 >= np.nanmax(ys)) for xs, ys in zip(xvals, yvals)])