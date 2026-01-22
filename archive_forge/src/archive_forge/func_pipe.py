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
@doc(klass=_shared_doc_kwargs['klass'])
def pipe(self, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs) -> T:
    """
        Apply chainable functions that expect Series or DataFrames.

        Parameters
        ----------
        func : function
            Function to apply to the {klass}.
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the {klass}.
        *args : iterable, optional
            Positional arguments passed into ``func``.
        **kwargs : mapping, optional
            A dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        the return type of ``func``.

        See Also
        --------
        DataFrame.apply : Apply a function along input axis of DataFrame.
        DataFrame.map : Apply a function elementwise on a whole DataFrame.
        Series.map : Apply a mapping correspondence on a
            :class:`~pandas.Series`.

        Notes
        -----
        Use ``.pipe`` when chaining together functions that expect
        Series, DataFrames or GroupBy objects.

        Examples
        --------
        Constructing a income DataFrame from a dictionary.

        >>> data = [[8000, 1000], [9500, np.nan], [5000, 2000]]
        >>> df = pd.DataFrame(data, columns=['Salary', 'Others'])
        >>> df
           Salary  Others
        0    8000  1000.0
        1    9500     NaN
        2    5000  2000.0

        Functions that perform tax reductions on an income DataFrame.

        >>> def subtract_federal_tax(df):
        ...     return df * 0.9
        >>> def subtract_state_tax(df, rate):
        ...     return df * (1 - rate)
        >>> def subtract_national_insurance(df, rate, rate_increase):
        ...     new_rate = rate + rate_increase
        ...     return df * (1 - new_rate)

        Instead of writing

        >>> subtract_national_insurance(
        ...     subtract_state_tax(subtract_federal_tax(df), rate=0.12),
        ...     rate=0.05,
        ...     rate_increase=0.02)  # doctest: +SKIP

        You can write

        >>> (
        ...     df.pipe(subtract_federal_tax)
        ...     .pipe(subtract_state_tax, rate=0.12)
        ...     .pipe(subtract_national_insurance, rate=0.05, rate_increase=0.02)
        ... )
            Salary   Others
        0  5892.48   736.56
        1  6997.32      NaN
        2  3682.80  1473.12

        If you have a function that takes the data as (say) the second
        argument, pass a tuple indicating which keyword expects the
        data. For example, suppose ``national_insurance`` takes its data as ``df``
        in the second argument:

        >>> def subtract_national_insurance(rate, df, rate_increase):
        ...     new_rate = rate + rate_increase
        ...     return df * (1 - new_rate)
        >>> (
        ...     df.pipe(subtract_federal_tax)
        ...     .pipe(subtract_state_tax, rate=0.12)
        ...     .pipe(
        ...         (subtract_national_insurance, 'df'),
        ...         rate=0.05,
        ...         rate_increase=0.02
        ...     )
        ... )
            Salary   Others
        0  5892.48   736.56
        1  6997.32      NaN
        2  3682.80  1473.12
        """
    if using_copy_on_write():
        return common.pipe(self.copy(deep=None), func, *args, **kwargs)
    return common.pipe(self, func, *args, **kwargs)