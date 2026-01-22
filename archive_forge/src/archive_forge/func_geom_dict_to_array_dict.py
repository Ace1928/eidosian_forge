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
def geom_dict_to_array_dict(geom_dict, coord_names=None):
    """
    Converts a dictionary containing an geometry key to a dictionary
    of x- and y-coordinate arrays and if present a list-of-lists of
    hole array.
    """
    if coord_names is None:
        coord_names = ['Longitude', 'Latitude']
    x, y = coord_names
    geom = geom_dict['geometry']
    new_dict = {k: v for k, v in geom_dict.items() if k != 'geometry'}
    array = geom_to_array(geom)
    new_dict[x] = array[:, 0]
    new_dict[y] = array[:, 1]
    if geom.geom_type == 'Polygon':
        holes = []
        for interior in geom.interiors:
            holes.append(geom_to_array(interior))
        if holes:
            new_dict['holes'] = [holes]
    elif geom.geom_type == 'MultiPolygon':
        outer_holes = []
        for g in geom.geoms:
            holes = []
            for interior in g.interiors:
                holes.append(geom_to_array(interior))
            outer_holes.append(holes)
        if any((hs for hs in outer_holes)):
            new_dict['holes'] = outer_holes
    return new_dict