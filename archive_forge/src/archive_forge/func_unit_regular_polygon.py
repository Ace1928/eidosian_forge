import copy
from functools import lru_cache
from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment
@classmethod
def unit_regular_polygon(cls, numVertices):
    """
        Return a :class:`Path` instance for a unit regular polygon with the
        given *numVertices* such that the circumscribing circle has radius 1.0,
        centered at (0, 0).
        """
    if numVertices <= 16:
        path = cls._unit_regular_polygons.get(numVertices)
    else:
        path = None
    if path is None:
        theta = 2 * np.pi / numVertices * np.arange(numVertices + 1) + np.pi / 2
        verts = np.column_stack((np.cos(theta), np.sin(theta)))
        path = cls(verts, closed=True, readonly=True)
        if numVertices <= 16:
            cls._unit_regular_polygons[numVertices] = path
    return path