import copy
from functools import lru_cache
from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment
@classmethod
def unit_rectangle(cls):
    """
        Return a `Path` instance of the unit rectangle from (0, 0) to (1, 1).
        """
    if cls._unit_rectangle is None:
        cls._unit_rectangle = cls([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], closed=True, readonly=True)
    return cls._unit_rectangle