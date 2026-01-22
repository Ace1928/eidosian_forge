from __future__ import annotations
import collections
import re
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any
import numpy as np
import scipy.constants as const
def unitized(unit):
    """Useful decorator to assign units to the output of a function. You can also
    use it to standardize the output units of a function that already returns
    a FloatWithUnit or ArrayWithUnit. For sequences, all values in the sequences
    are assigned the same unit. It works with Python sequences only. The creation
    of numpy arrays loses all unit information. For mapping types, the values
    are assigned units.

    Args:
        unit: Specific unit (eV, Ha, m, ang, etc.).

    Example:
        @unitized(unit="kg")
        def get_mass():
            return 123.45
    """

    def wrap(func):

        def wrapped_f(*args, **kwargs):
            val = func(*args, **kwargs)
            unit_type = _UNAME2UTYPE[unit]
            if isinstance(val, (FloatWithUnit, ArrayWithUnit)):
                return val.to(unit)
            if isinstance(val, collections.abc.Sequence):
                return val.__class__([FloatWithUnit(i, unit_type=unit_type, unit=unit) for i in val])
            if isinstance(val, collections.abc.Mapping):
                for k, v in val.items():
                    val[k] = FloatWithUnit(v, unit_type=unit_type, unit=unit)
            elif isinstance(val, Number):
                return FloatWithUnit(val, unit_type=unit_type, unit=unit)
            elif val is None:
                pass
            else:
                raise TypeError(f"Don't know how to assign units to {val}")
            return val
        return wrapped_f
    return wrap