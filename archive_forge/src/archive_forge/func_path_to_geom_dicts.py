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
def path_to_geom_dicts(fullpath, skip_invalid=True):
    """
    Converts a Path element into a list of geometry dictionaries,
    preserving all value dimensions.
    """
    geoms = unpack_geoms(fullpath)
    if geoms is not None:
        return geoms
    geoms = []
    invalid = False
    xdim, ydim = fullpath.kdims
    for path in fullpath.split(datatype='columns'):
        array = np.column_stack([path.pop(xdim.name), path.pop(ydim.name)])
        splits = np.where(np.isnan(array[:, :2].astype('float')).sum(axis=1))[0]
        arrays = np.split(array, splits + 1) if len(splits) else [array]
        subpaths = []
        for j, arr in enumerate(arrays):
            if j != len(arrays) - 1:
                arr = arr[:-1]
            if len(arr) == 0:
                continue
            elif len(arr) == 1:
                if skip_invalid:
                    continue
                g = Point(arr[0])
                invalid = True
            else:
                g = LineString(arr)
            subpaths.append(g)
        if invalid:
            geoms += [dict(path, geometry=sp) for sp in subpaths]
            continue
        elif len(subpaths) == 1:
            geom = subpaths[0]
        elif subpaths:
            geom = MultiLineString(subpaths)
        else:
            continue
        path['geometry'] = geom
        geoms.append(path)
    return geoms