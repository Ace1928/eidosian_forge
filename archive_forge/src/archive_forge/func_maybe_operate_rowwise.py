from __future__ import annotations
import functools
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
def maybe_operate_rowwise(func: F) -> F:
    """
    NumPy operations on C-contiguous ndarrays with axis=1 can be
    very slow if axis 1 >> axis 0.
    Operate row-by-row and concatenate the results.
    """

    @functools.wraps(func)
    def newfunc(values: np.ndarray, *, axis: AxisInt | None=None, **kwargs):
        if axis == 1 and values.ndim == 2 and values.flags['C_CONTIGUOUS'] and (values.shape[1] / 1000 > values.shape[0]) and (values.dtype != object) and (values.dtype != bool):
            arrs = list(values)
            if kwargs.get('mask') is not None:
                mask = kwargs.pop('mask')
                results = [func(arrs[i], mask=mask[i], **kwargs) for i in range(len(arrs))]
            else:
                results = [func(x, **kwargs) for x in arrs]
            return np.array(results)
        return func(values, axis=axis, **kwargs)
    return cast(F, newfunc)