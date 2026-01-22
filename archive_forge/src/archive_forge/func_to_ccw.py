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
def to_ccw(geom):
    """
    Reorients polygon to be wound counter-clockwise.
    """
    if isinstance(geom, sgeom.Polygon) and (not geom.exterior.is_ccw):
        geom = sgeom.polygon.orient(geom)
    return geom