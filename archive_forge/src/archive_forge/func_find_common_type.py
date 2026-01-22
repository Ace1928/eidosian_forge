from __future__ import annotations
import datetime as dt
import functools
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import (
from pandas._libs.missing import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.compat.numpy import np_version_gt2
from pandas.errors import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
from pandas.core.dtypes.missing import (
from pandas.io._util import _arrow_dtype_mapping
def find_common_type(types):
    """
    Find a common data type among the given dtypes.

    Parameters
    ----------
    types : list of dtypes

    Returns
    -------
    pandas extension or numpy dtype

    See Also
    --------
    numpy.find_common_type

    """
    if not types:
        raise ValueError('no types given')
    first = types[0]
    if lib.dtypes_all_equal(list(types)):
        return first
    types = list(dict.fromkeys(types).keys())
    if any((isinstance(t, ExtensionDtype) for t in types)):
        for t in types:
            if isinstance(t, ExtensionDtype):
                res = t._get_common_dtype(types)
                if res is not None:
                    return res
        return np.dtype('object')
    if all((lib.is_np_dtype(t, 'M') for t in types)):
        return np.dtype(max(types))
    if all((lib.is_np_dtype(t, 'm') for t in types)):
        return np.dtype(max(types))
    has_bools = any((t.kind == 'b' for t in types))
    if has_bools:
        for t in types:
            if t.kind in 'iufc':
                return np.dtype('object')
    return np_find_common_type(*types)