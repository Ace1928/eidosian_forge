import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from ..dimension import dimension_name
from ..util import isscalar, unique_array, unique_iterator
from .interface import DataError, Interface
from .multipath import MultiInterface, ensure_ring
from .pandas import PandasInterface
def geom_to_holes(geom):
    """Extracts holes from spatialpandas Polygon geometries.

    Args:
        geom: spatialpandas geometry

    Returns:
        List of arrays representing holes
    """
    from spatialpandas.geometry import MultiPolygon, Polygon
    if isinstance(geom, Polygon):
        holes = []
        for i, hole in enumerate(geom.data):
            if i == 0:
                continue
            hole = ensure_ring(np.array(hole.as_py()).reshape(-1, 2))
            holes.append(hole)
        return [holes]
    elif isinstance(geom, MultiPolygon):
        holes = []
        for poly in geom.data:
            poly_holes = []
            for i, hole in enumerate(poly):
                if i == 0:
                    continue
                arr = ensure_ring(np.array(hole.as_py()).reshape(-1, 2))
                poly_holes.append(arr)
            holes.append(poly_holes)
        return holes
    elif 'Multi' in type(geom).__name__:
        return [[]] * len(geom)
    else:
        return [[]]