from __future__ import annotations
import dataclasses
import datetime
import decimal
import operator
import pathlib
import pickle
import random
import subprocess
import sys
import textwrap
from enum import Enum, Flag, IntEnum, IntFlag
from typing import Union
import cloudpickle
import pytest
from tlz import compose, curry, partial
import dask
from dask.base import TokenizationError, normalize_token, tokenize
from dask.core import literal
from dask.utils import tmpfile
from dask.utils_test import import_or_none
@pytest.mark.skipif('not dd')
def test_tokenize_pandas_extension_array():
    arrays = [pd.array([1, 0, None], dtype='Int64'), pd.array(['2000'], dtype='Period[D]'), pd.array([1, 0, 0], dtype='Sparse[int]'), pd.array([pd.Timestamp('2000')], dtype='datetime64[ns]'), pd.array([pd.Timestamp('2000', tz='CET')], dtype='datetime64[ns, CET]'), pd.array(['a', 'b'], dtype=pd.api.types.CategoricalDtype(['a', 'b', 'c'], ordered=False))]
    arrays.extend([pd.array(['a', 'b', None], dtype='string'), pd.array([True, False, None], dtype='boolean')])
    for arr in arrays:
        check_tokenize(arr)