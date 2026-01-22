from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.internals import BlockPlacement
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import (
from pandas.core.arrays import DatetimeArray
from pandas.core.construction import extract_array
from pandas.core.internals.blocks import (
def maybe_infer_ndim(values, placement: BlockPlacement, ndim: int | None) -> int:
    """
    If `ndim` is not provided, infer it from placement and values.
    """
    if ndim is None:
        if not isinstance(values.dtype, np.dtype):
            if len(placement) != 1:
                ndim = 1
            else:
                ndim = 2
        else:
            ndim = values.ndim
    return ndim