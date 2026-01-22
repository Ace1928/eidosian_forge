import sys
from collections.abc import Hashable
from functools import wraps
from packaging.version import Version
from types import FunctionType
import bokeh
import numpy as np
import pandas as pd
import param
import holoviews as hv
def process_crs(crs):
    """
    Parses cartopy CRS definitions defined in one of a few formats:

      1. EPSG codes:   Defined as string of the form "EPSG: {code}" or an integer
      2. proj.4 string: Defined as string of the form "{proj.4 string}"
      3. cartopy.crs.CRS instance
      3. pyproj.Proj or pyproj.CRS instance
      4. WKT string:    Defined as string of the form "{WKT string}"
      5. None defaults to crs.PlateCaree
    """
    missing = []
    try:
        import cartopy.crs as ccrs
    except ImportError:
        missing.append('cartopy')
    try:
        import geoviews as gv
    except ImportError:
        missing.append('geoviews')
    try:
        import pyproj
    except ImportError:
        missing.append('pyproj')
    if missing:
        raise ImportError(f'Geographic projection support requires: {', '.join(missing)}.')
    if crs is None:
        return ccrs.PlateCarree()
    elif isinstance(crs, ccrs.CRS):
        return crs
    elif isinstance(crs, pyproj.CRS):
        crs = crs.to_wkt()
    errors = []
    if isinstance(crs, (str, int)):
        try:
            crs = pyproj.CRS.from_epsg(crs).to_wkt()
        except Exception as e:
            errors.append(e)
    if isinstance(crs, (str, pyproj.Proj)):
        try:
            return proj_to_cartopy(crs)
        except Exception as e:
            errors.append(e)
    raise ValueError('Projection must be defined as a EPSG code, proj4 string, WKT string, cartopy CRS, pyproj.Proj, or pyproj.CRS.') from Exception(*errors)