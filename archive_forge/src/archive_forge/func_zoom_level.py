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
def zoom_level(bounds, width, height):
    """
    Compute zoom level given bounds and the plot size.
    """
    w, s, e, n = bounds
    max_width, max_height = (256, 256)
    ZOOM_MAX = 21
    ln2 = np.log(2)

    def latRad(lat):
        sin = np.sin(lat * np.pi / 180)
        radX2 = np.log((1 + sin) / (1 - sin)) / 2
        return np.max([np.min([radX2, np.pi]), -np.pi]) / 2

    def zoom(mapPx, worldPx, fraction):
        return np.floor(np.log(mapPx / worldPx / fraction) / ln2)
    latFraction = (latRad(n) - latRad(s)) / np.pi
    lngDiff = e - w
    lngFraction = (lngDiff + 360 if lngDiff < 0 else lngDiff) / 360
    latZoom = zoom(height, max_height, latFraction)
    lngZoom = zoom(width, max_width, lngFraction)
    zoom = np.min([latZoom, lngZoom, ZOOM_MAX])
    return int(zoom) if np.isfinite(zoom) else 0