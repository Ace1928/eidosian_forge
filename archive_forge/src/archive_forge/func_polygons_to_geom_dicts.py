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
def polygons_to_geom_dicts(polygons, skip_invalid=True):
    """
    Converts a Polygons element into a list of geometry dictionaries,
    preserving all value dimensions.

    For array conversion the following conventions are applied:

    * Any nan separated array are converted into a MultiPolygon
    * Any array without nans is converted to a Polygon
    * If there are holes associated with a nan separated array
      the holes are assigned to the polygons by testing for an
      intersection
    * If any single array does not have at least three coordinates
      it is skipped by default
    * If skip_invalid=False and an array has less than three
      coordinates it will be converted to a LineString
    """
    geoms = unpack_geoms(polygons)
    if geoms is not None:
        return geoms
    polys = []
    xdim, ydim = polygons.kdims
    has_holes = polygons.has_holes
    holes = polygons.holes() if has_holes else None
    for i, polygon in enumerate(polygons.split(datatype='columns')):
        array = np.column_stack([polygon.pop(xdim.name), polygon.pop(ydim.name)])
        splits = np.where(np.isnan(array[:, :2].astype('float')).sum(axis=1))[0]
        arrays = np.split(array, splits + 1) if len(splits) else [array]
        invalid = False
        subpolys = []
        subholes = None
        if has_holes:
            subholes = [[LinearRing(h) for h in hs] for hs in holes[i]]
        for j, arr in enumerate(arrays):
            if j != len(arrays) - 1:
                arr = arr[:-1]
            if len(arr) == 0:
                continue
            elif len(arr) == 1:
                if skip_invalid:
                    continue
                poly = Point(arr[0])
                invalid = True
            elif len(arr) == 2:
                if skip_invalid:
                    continue
                poly = LineString(arr)
                invalid = True
            elif not len(splits):
                poly = Polygon(arr, subholes[j] if has_holes else [])
            else:
                poly = Polygon(arr)
                hs = [h for h in subholes[j]] if has_holes else []
                poly = Polygon(poly.exterior, holes=hs)
            subpolys.append(poly)
        if invalid:
            polys += [dict(polygon, geometry=sp) for sp in subpolys]
            continue
        elif len(subpolys) == 1:
            geom = subpolys[0]
        elif subpolys:
            geom = MultiPolygon(subpolys)
        else:
            continue
        polygon['geometry'] = geom
        polys.append(polygon)
    return polys