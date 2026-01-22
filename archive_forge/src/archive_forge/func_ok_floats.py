from __future__ import annotations
import warnings
from platform import machine, processor
import numpy as np
from .deprecated import deprecate_with_version
def ok_floats():
    """Return floating point types sorted by precision

    Remove longdouble if it has no higher precision than float64
    """
    floats = sctypes['float'][:]
    if best_float() != np.longdouble and np.longdouble in floats:
        floats.remove(np.longdouble)
    return sorted(floats, key=lambda f: type_info(f)['nmant'])