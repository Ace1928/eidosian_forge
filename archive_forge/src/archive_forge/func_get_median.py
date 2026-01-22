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
def get_median(x, _mask=None):
    if _mask is None:
        _mask = notna(x)
    else:
        _mask = ~_mask
    if not skipna and (not _mask.all()):
        return np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
        res = np.nanmedian(x[_mask])
    return res