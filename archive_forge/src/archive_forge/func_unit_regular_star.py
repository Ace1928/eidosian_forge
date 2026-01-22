import copy
from functools import lru_cache
from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment
@classmethod
def unit_regular_star(cls, numVertices, innerCircle=0.5):
    """
        Return a :class:`Path` for a unit regular star with the given
        numVertices and radius of 1.0, centered at (0, 0).
        """
    if numVertices <= 16:
        path = cls._unit_regular_stars.get((numVertices, innerCircle))
    else:
        path = None
    if path is None:
        ns2 = numVertices * 2
        theta = 2 * np.pi / ns2 * np.arange(ns2 + 1)
        theta += np.pi / 2.0
        r = np.ones(ns2 + 1)
        r[1::2] = innerCircle
        verts = (r * np.vstack((np.cos(theta), np.sin(theta)))).T
        path = cls(verts, closed=True, readonly=True)
        if numVertices <= 16:
            cls._unit_regular_stars[numVertices, innerCircle] = path
    return path