from __future__ import annotations
from contextlib import contextmanager
import operator
import numba
from numba import types
from numba.core import cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
import numpy as np
from pandas._libs import lib
from pandas.core.indexes.base import Index
from pandas.core.indexing import _iLocIndexer
from pandas.core.internals import SingleBlockManager
from pandas.core.series import Series
@contextmanager
def set_numba_data(index: Index):
    numba_data = index._data
    if numba_data.dtype == object:
        if not lib.is_string_array(numba_data):
            raise ValueError('The numba engine only supports using string or numeric column names')
        numba_data = numba_data.astype('U')
    try:
        index._numba_data = numba_data
        yield index
    finally:
        del index._numba_data