import copy
from functools import lru_cache
from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment
@staticmethod
@lru_cache(8)
def hatch(hatchpattern, density=6):
    """
        Given a hatch specifier, *hatchpattern*, generates a `Path` that
        can be used in a repeated hatching pattern.  *density* is the
        number of lines per unit square.
        """
    from matplotlib.hatch import get_path
    return get_path(hatchpattern, density) if hatchpattern is not None else None