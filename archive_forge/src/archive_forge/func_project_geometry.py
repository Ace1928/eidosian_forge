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
def project_geometry(self, geometry, src_crs=None):
    """
        Project the given geometry into this projection.

        Parameters
        ----------
        geometry
            The geometry to (re-)project.
        src_crs: optional
            The source CRS.  Defaults to None.

            If src_crs is None, the source CRS is assumed to be a geodetic
            version of the target CRS.

        Returns
        -------
        geometry
            The projected result (a shapely geometry).

        """
    if src_crs is None:
        src_crs = self.as_geodetic()
    elif not isinstance(src_crs, CRS):
        raise TypeError('Source CRS must be an instance of CRS or one of its subclasses, or None.')
    geom_type = geometry.geom_type
    method_name = self._method_map.get(geom_type)
    if not method_name:
        raise ValueError(f'Unsupported geometry type {geom_type!r}')
    return getattr(self, method_name)(geometry, src_crs)