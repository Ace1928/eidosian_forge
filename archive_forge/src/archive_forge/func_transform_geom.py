from math import ceil, floor
from affine import Affine
import numpy as np
import rasterio
from rasterio._base import _transform
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.env import ensure_env, require_gdal_version
from rasterio.errors import TransformError, RPCError
from rasterio.transform import array_bounds
from rasterio._warp import (
@ensure_env
@require_gdal_version('2.1', param='antimeridian_cutting', values=[False], is_max_version=True, reason='Antimeridian cutting is always enabled on GDAL >= 2.2')
def transform_geom(src_crs, dst_crs, geom, antimeridian_cutting=True, antimeridian_offset=10.0, precision=-1):
    """Transform geometry from source coordinate reference system into target.

    Parameters
    ------------
    src_crs: CRS or dict
        Source coordinate reference system, in rasterio dict format.
        Example: CRS({'init': 'EPSG:4326'})
    dst_crs: CRS or dict
        Target coordinate reference system.
    geom: GeoJSON like dict object or iterable of GeoJSON like objects.
    antimeridian_cutting: bool, optional
        If True, cut geometries at the antimeridian, otherwise geometries
        will not be cut (default).  If False and GDAL is 2.2.0 or newer
        an exception is raised.  Antimeridian cutting is always on as of
        GDAL 2.2.0 but this could produce an unexpected geometry.
    antimeridian_offset: float
        Offset from the antimeridian in degrees (default: 10) within which
        any geometries will be split.
    precision: float
        If >= 0, geometry coordinates will be rounded to this number of decimal
        places after the transform operation, otherwise original coordinate
        values will be preserved (default).

    Returns
    ---------
    out: GeoJSON like dict object or list of GeoJSON like objects.
        Transformed geometry(s) in GeoJSON dict format
    """
    return _transform_geom(src_crs, dst_crs, geom, antimeridian_cutting, antimeridian_offset, precision)