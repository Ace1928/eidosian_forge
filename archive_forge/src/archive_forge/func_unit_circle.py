import copy
from functools import lru_cache
from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment
@classmethod
def unit_circle(cls):
    """
        Return the readonly :class:`Path` of the unit circle.

        For most cases, :func:`Path.circle` will be what you want.
        """
    if cls._unit_circle is None:
        cls._unit_circle = cls.circle(center=(0, 0), radius=1, readonly=True)
    return cls._unit_circle