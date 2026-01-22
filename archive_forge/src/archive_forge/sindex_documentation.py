import warnings
from shapely.geometry.base import BaseGeometry
import pandas as pd
import numpy as np
from . import _compat as compat
from ._decorator import doc
Convert geometry into a numpy array of PyGEOS geometries.

            Parameters
            ----------
            geometry
                An array-like of PyGEOS geometries, a GeoPandas GeoSeries/GeometryArray,
                shapely.geometry or list of shapely geometries.

            Returns
            -------
            np.ndarray
                A numpy array of pygeos geometries.
            