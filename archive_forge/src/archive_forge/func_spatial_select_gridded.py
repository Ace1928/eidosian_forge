import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
def spatial_select_gridded(xvals, yvals, geometry):
    rectilinear = (np.diff(xvals, axis=0) == 0).all()
    if rectilinear:
        from .path import Polygons
        from .raster import Image
        try:
            from ..operation.datashader import rasterize
        except ImportError:
            raise ImportError('Lasso selection on gridded data requires datashader to be available.') from None
        xs, ys = (xvals[0], yvals[:, 0])
        target = Image((xs, ys, np.empty(ys.shape + xs.shape)))
        poly = Polygons([geometry])
        sel_mask = rasterize(poly, target=target, dynamic=False, aggregator='any')
        return sel_mask.dimension_values(2, flat=False)
    else:
        sel_mask = spatial_select_columnar(xvals.flatten(), yvals.flatten(), geometry)
        return sel_mask.reshape(xvals.shape)