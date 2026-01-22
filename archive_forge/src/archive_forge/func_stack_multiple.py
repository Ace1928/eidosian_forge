from __future__ import annotations
import itertools
from typing import (
import warnings
import numpy as np
import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.algorithms import (
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import (
def stack_multiple(frame: DataFrame, level, dropna: bool=True, sort: bool=True):
    if all((lev in frame.columns.names for lev in level)):
        result = frame
        for lev in level:
            result = stack(result, lev, dropna=dropna, sort=sort)
    elif all((isinstance(lev, int) for lev in level)):
        result = frame
        level = [frame.columns._get_level_number(lev) for lev in level]
        while level:
            lev = level.pop(0)
            result = stack(result, lev, dropna=dropna, sort=sort)
            level = [v if v <= lev else v - 1 for v in level]
    else:
        raise ValueError('level should contain all level names or all level numbers, not a mixture of the two.')
    return result