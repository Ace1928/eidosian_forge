from __future__ import annotations
import collections
from copy import deepcopy
import datetime as dt
from functools import partial
import gc
from json import loads
import operator
import pickle
import re
import sys
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.lib import is_range_indexer
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.array_algos.replace import should_use_regex
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.flags import Flags
from pandas.core.indexes.api import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods.describe import describe_ndframe
from pandas.core.missing import (
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import (
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
@final
def first(self, offset) -> Self:
    """
        Select initial periods of time series data based on a date offset.

        .. deprecated:: 2.1
            :meth:`.first` is deprecated and will be removed in a future version.
            Please create a mask and filter using `.loc` instead.

        For a DataFrame with a sorted DatetimeIndex, this function can
        select the first few rows based on a date offset.

        Parameters
        ----------
        offset : str, DateOffset or dateutil.relativedelta
            The offset length of the data that will be selected. For instance,
            '1ME' will display all the rows having their index within the first month.

        Returns
        -------
        Series or DataFrame
            A subset of the caller.

        Raises
        ------
        TypeError
            If the index is not  a :class:`DatetimeIndex`

        See Also
        --------
        last : Select final periods of time series based on a date offset.
        at_time : Select values at a particular time of the day.
        between_time : Select values between particular times of the day.

        Examples
        --------
        >>> i = pd.date_range('2018-04-09', periods=4, freq='2D')
        >>> ts = pd.DataFrame({'A': [1, 2, 3, 4]}, index=i)
        >>> ts
                    A
        2018-04-09  1
        2018-04-11  2
        2018-04-13  3
        2018-04-15  4

        Get the rows for the first 3 days:

        >>> ts.first('3D')
                    A
        2018-04-09  1
        2018-04-11  2

        Notice the data for 3 first calendar days were returned, not the first
        3 days observed in the dataset, and therefore data for 2018-04-13 was
        not returned.
        """
    warnings.warn('first is deprecated and will be removed in a future version. Please create a mask and filter using `.loc` instead', FutureWarning, stacklevel=find_stack_level())
    if not isinstance(self.index, DatetimeIndex):
        raise TypeError("'first' only supports a DatetimeIndex index")
    if len(self.index) == 0:
        return self.copy(deep=False)
    offset = to_offset(offset)
    if not isinstance(offset, Tick) and offset.is_on_offset(self.index[0]):
        end_date = end = self.index[0] - offset.base + offset
    else:
        end_date = end = self.index[0] + offset
    if isinstance(offset, Tick) and end_date in self.index:
        end = self.index.searchsorted(end_date, side='left')
        return self.iloc[:end]
    return self.loc[:end]