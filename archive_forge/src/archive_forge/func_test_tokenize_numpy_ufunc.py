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
@pytest.mark.skipif('not np')
def test_tokenize_numpy_ufunc():
    assert check_tokenize(np.sin) != check_tokenize(np.cos)
    np_ufunc = np.sin
    np_ufunc2 = np.cos
    assert isinstance(np_ufunc, np.ufunc)
    assert isinstance(np_ufunc2, np.ufunc)
    assert check_tokenize(np_ufunc) != check_tokenize(np_ufunc2)
    inc = da.ufunc.frompyfunc(lambda x: x + 1, 1, 1)
    inc2 = da.ufunc.frompyfunc(lambda x: x + 1, 1, 1)
    inc3 = da.ufunc.frompyfunc(lambda x: x + 2, 1, 1)
    assert check_tokenize(inc) != check_tokenize(inc2)
    assert check_tokenize(inc) != check_tokenize(inc3)