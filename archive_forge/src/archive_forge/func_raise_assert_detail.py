from __future__ import annotations
import operator
from typing import (
import numpy as np
from pandas._libs import lib
from pandas._libs.missing import is_matching_na
from pandas._libs.sparse import SparseIndex
import pandas._libs.testing as _testing
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
from pandas import (
from pandas.core.arrays import (
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
from pandas.core.arrays.string_ import StringDtype
from pandas.core.indexes.api import safe_sort_index
from pandas.io.formats.printing import pprint_thing
def raise_assert_detail(obj, message, left, right, diff=None, first_diff=None, index_values=None) -> NoReturn:
    __tracebackhide__ = True
    msg = f'{obj} are different\n\n{message}'
    if isinstance(index_values, Index):
        index_values = np.asarray(index_values)
    if isinstance(index_values, np.ndarray):
        msg += f'\n[index]: {pprint_thing(index_values)}'
    if isinstance(left, np.ndarray):
        left = pprint_thing(left)
    elif isinstance(left, (CategoricalDtype, NumpyEADtype, StringDtype)):
        left = repr(left)
    if isinstance(right, np.ndarray):
        right = pprint_thing(right)
    elif isinstance(right, (CategoricalDtype, NumpyEADtype, StringDtype)):
        right = repr(right)
    msg += f'\n[left]:  {left}\n[right]: {right}'
    if diff is not None:
        msg += f'\n[diff]: {diff}'
    if first_diff is not None:
        msg += f'\n{first_diff}'
    raise AssertionError(msg)