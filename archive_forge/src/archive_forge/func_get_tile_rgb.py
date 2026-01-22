import numpy as np
import shapely
import shapely.geometry as sgeom
from cartopy import crs as ccrs
from cartopy.io.img_tiles import GoogleTiles, QuadtreeTiles
from holoviews.element import Tiles
from packaging.version import Version
from shapely.geometry import (
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import transform
from ._warnings import warn
def get_tile_rgb(tile_source, bbox, zoom_level, bbox_crs=None):
    """
    Returns an RGB element given a tile_source, bounding box and zoom level.

    Parameters
    ----------
    tile_source: WMTS element or string URL
      The tile source to download the tiles from.
    bbox: tuple
      A four tuple specifying the (left, bottom, right, top) corners of the
      domain to download the tiles for.
    zoom_level: int
      The zoom level at which to download the tiles
    bbox_crs: ccrs.CRs
      cartopy CRS defining the coordinate system of the supplied bbox

    Returns
    -------
    RGB element containing the tile data in the specified bbox
    """
    from .element import RGB, WMTS
    if bbox_crs is None:
        bbox_crs = ccrs.PlateCarree()
    if isinstance(tile_source, (WMTS, Tiles)):
        tile_source = tile_source.data
    if bbox_crs is not ccrs.GOOGLE_MERCATOR:
        bbox = project_extents(bbox, bbox_crs, ccrs.GOOGLE_MERCATOR)
    if '{Q}' in tile_source:
        tile_source = QuadtreeTiles(url=tile_source.replace('{Q}', '{tile}'))
    else:
        tile_source = GoogleTiles(url=tile_source)
    bounds = box(*bbox)
    rgb, extent, orient = tile_source.image_for_domain(bounds, zoom_level)
    if orient == 'lower':
        rgb = rgb[::-1]
    x0, x1, y0, y1 = extent
    l, b, r, t = bbox
    return RGB(rgb, bounds=(x0, y0, x1, y1), crs=ccrs.GOOGLE_MERCATOR, vdims=['R', 'G', 'B']).clone(datatype=['grid', 'xarray', 'iris'])[l:r, b:t]