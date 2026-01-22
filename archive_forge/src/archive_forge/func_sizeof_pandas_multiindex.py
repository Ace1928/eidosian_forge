from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
@sizeof.register(pd.MultiIndex)
def sizeof_pandas_multiindex(i):
    return sum((sizeof(l) for l in i.levels)) + sum((c.nbytes for c in i.codes))