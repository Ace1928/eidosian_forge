from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
@sizeof.register(pd.Series)
def sizeof_pandas_series(s):
    p = 1200 + sizeof(s.index) + s.memory_usage(index=False, deep=False)
    if s.dtype in OBJECT_DTYPES:
        p += object_size(s._values)
    return p