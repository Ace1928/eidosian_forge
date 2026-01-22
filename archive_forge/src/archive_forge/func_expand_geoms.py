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
def expand_geoms(geoms):
    """
    Expands multi-part geometries in a list of geometries.
    """
    expanded = []
    for geom in geoms:
        if isinstance(geom, BaseMultipartGeometry):
            expanded.extend(list(geom))
        else:
            expanded.append(geom)
    return expanded