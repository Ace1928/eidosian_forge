from __future__ import annotations
from functools import (
from typing import (
import warnings
import numpy as np
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.generic import (
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation.common import result_type_many
def reconstruct_object(typ, obj, axes, dtype):
    """
    Reconstruct an object given its type, raw value, and possibly empty
    (None) axes.

    Parameters
    ----------
    typ : object
        A type
    obj : object
        The value to use in the type constructor
    axes : dict
        The axes to use to construct the resulting pandas object

    Returns
    -------
    ret : typ
        An object of type ``typ`` with the value `obj` and possible axes
        `axes`.
    """
    try:
        typ = typ.type
    except AttributeError:
        pass
    res_t = np.result_type(obj.dtype, dtype)
    if not isinstance(typ, partial) and issubclass(typ, PandasObject):
        return typ(obj, dtype=res_t, **axes)
    if hasattr(res_t, 'type') and typ == np.bool_ and (res_t != np.bool_):
        ret_value = res_t.type(obj)
    else:
        ret_value = typ(obj).astype(res_t)
        if len(obj.shape) == 1 and len(obj) == 1 and (not isinstance(ret_value, np.ndarray)):
            ret_value = np.array([ret_value]).astype(res_t)
    return ret_value