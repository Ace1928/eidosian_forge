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
@property
@Appender(_interval_shared_docs['is_non_overlapping_monotonic'] % _shared_docs_kwargs)
def is_non_overlapping_monotonic(self) -> bool:
    if self.closed == 'both':
        return bool((self._right[:-1] < self._left[1:]).all() or (self._left[:-1] > self._right[1:]).all())
    return bool((self._right[:-1] <= self._left[1:]).all() or (self._left[:-1] >= self._right[1:]).all())