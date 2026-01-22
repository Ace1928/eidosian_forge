from __future__ import annotations
import collections
import itertools as it
import operator
import uuid
import warnings
from functools import partial, wraps
from numbers import Integral
import numpy as np
import pandas as pd
from dask.base import is_dask_collection, tokenize
from dask.core import flatten
from dask.dataframe._compat import (
from dask.dataframe.core import (
from dask.dataframe.dispatch import grouper_dispatch
from dask.dataframe.methods import concat, drop_columns
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.typing import no_default
from dask.utils import (
def numeric_only_not_implemented(func):
    """Decorator for methods that can't handle numeric_only=False"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if isinstance(self, DataFrameGroupBy):
            maybe_raise = not (func.__name__ == 'agg' and len(args) > 0 and (args[0] not in NUMERIC_ONLY_NOT_IMPLEMENTED))
            if maybe_raise:
                numeric_only = kwargs.get('numeric_only', no_default)
                if not PANDAS_GE_150 and numeric_only is False:
                    raise NotImplementedError("'numeric_only=False' is not implemented in Dask.")
                if not self._all_numeric():
                    if numeric_only is False or (PANDAS_GE_200 and numeric_only is no_default):
                        raise NotImplementedError("'numeric_only=False' is not implemented in Dask.")
                    if PANDAS_GE_150 and (not PANDAS_GE_200) and (numeric_only is no_default):
                        warnings.warn('The default value of numeric_only will be changed to False in the future when using dask with pandas 2.0', FutureWarning)
        return func(self, *args, **kwargs)
    return wrapper