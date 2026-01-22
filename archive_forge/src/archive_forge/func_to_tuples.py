from __future__ import annotations
import operator
from operator import (
import textwrap
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.interval import (
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import IntCastingNaNError
from pandas.util._decorators import Appender
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core.algorithms import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import (
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.timedeltas import TimedeltaArray
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import check_array_indexer
from pandas.core.ops import (
from_arrays
from_tuples
from_breaks
@Appender(_interval_shared_docs['to_tuples'] % {'return_type': 'ndarray (if self is IntervalArray) or Index (if self is IntervalIndex)', 'examples': textwrap.dedent("\n         Examples\n         --------\n         For :class:`pandas.IntervalArray`:\n\n         >>> idx = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])\n         >>> idx\n         <IntervalArray>\n         [(0, 1], (1, 2]]\n         Length: 2, dtype: interval[int64, right]\n         >>> idx.to_tuples()\n         array([(0, 1), (1, 2)], dtype=object)\n\n         For :class:`pandas.IntervalIndex`:\n\n         >>> idx = pd.interval_range(start=0, end=2)\n         >>> idx\n         IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')\n         >>> idx.to_tuples()\n         Index([(0, 1), (1, 2)], dtype='object')\n         ")})
def to_tuples(self, na_tuple: bool=True) -> np.ndarray:
    tuples = com.asarray_tuplesafe(zip(self._left, self._right))
    if not na_tuple:
        tuples = np.where(~self.isna(), tuples, np.nan)
    return tuples