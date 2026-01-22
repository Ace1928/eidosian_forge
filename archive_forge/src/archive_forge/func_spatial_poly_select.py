import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
def spatial_poly_select(xvals, yvals, geometry):
    try:
        from shapely.geometry import Polygon
        boxes = (Polygon(np.column_stack([xs, ys])) for xs, ys in zip(xvals, yvals))
        poly = Polygon(geometry)
        return np.array([poly.contains(p) for p in boxes])
    except ImportError:
        raise ImportError('Lasso selection on geometry data requires shapely to be available.') from None