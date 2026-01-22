from warnings import warn
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from shapely.geometry import box, MultiPoint
from shapely.geometry.base import BaseGeometry
from . import _compat as compat
from .array import GeometryArray, GeometryDtype, points_from_xy
def is_geometry_type(data):
    """
    Check if the data is of geometry dtype.

    Does not include object array of shapely scalars.
    """
    if isinstance(getattr(data, 'dtype', None), GeometryDtype):
        return True
    else:
        return False