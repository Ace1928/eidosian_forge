from abc import ABCMeta
from collections import OrderedDict
from functools import lru_cache
import io
import json
import math
import warnings
import numpy as np
from pyproj import Transformer
from pyproj.exceptions import ProjError
import shapely.geometry as sgeom
from shapely.prepared import prep
import cartopy.trace
def transform_points(self, src_crs, x, y, z=None, trap=False):
    """
        Capture and handle NaNs in input points -- else as parent function,
        :meth:`_WarpedRectangularProjection.transform_points`.

        Needed because input NaNs can trigger a fatal error in the underlying
        implementation of the Robinson projection.

        Note
        ----
            Although the original can in fact translate (nan, lat) into
            (nan, y-value), this patched version doesn't support that.
            Instead, we invalidate any of the points that contain a NaN.

        """
    input_point_nans = np.isnan(x) | np.isnan(y)
    if z is not None:
        input_point_nans |= np.isnan(z)
    handle_nans = np.any(input_point_nans)
    if handle_nans:
        x[input_point_nans] = 0.0
        y[input_point_nans] = 0.0
        if z is not None:
            z[input_point_nans] = 0.0
    result = super().transform_points(src_crs, x, y, z, trap=trap)
    if handle_nans:
        result[input_point_nans] = np.nan
    return result