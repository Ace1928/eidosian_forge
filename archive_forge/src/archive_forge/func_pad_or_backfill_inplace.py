from __future__ import annotations
from functools import wraps
from typing import (
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import (
def pad_or_backfill_inplace(values: np.ndarray, method: Literal['pad', 'backfill']='pad', axis: AxisInt=0, limit: int | None=None, limit_area: Literal['inside', 'outside'] | None=None) -> None:
    """
    Perform an actual interpolation of values, values will be make 2-d if
    needed fills inplace, returns the result.

    Parameters
    ----------
    values: np.ndarray
        Input array.
    method: str, default "pad"
        Interpolation method. Could be "bfill" or "pad"
    axis: 0 or 1
        Interpolation axis
    limit: int, optional
        Index limit on interpolation.
    limit_area: str, optional
        Limit area for interpolation. Can be "inside" or "outside"

    Notes
    -----
    Modifies values in-place.
    """
    transf = (lambda x: x) if axis == 0 else lambda x: x.T
    if values.ndim == 1:
        if axis != 0:
            raise AssertionError('cannot interpolate on a ndim == 1 with axis != 0')
        values = values.reshape(tuple((1,) + values.shape))
    method = clean_fill_method(method)
    tvalues = transf(values)
    func = get_fill_func(method, ndim=2)
    func(tvalues, limit=limit, limit_area=limit_area)