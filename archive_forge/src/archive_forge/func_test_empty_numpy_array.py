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
def test_empty_numpy_array():
    arr = np.array([])
    assert arr.strides
    assert check_tokenize(arr) == check_tokenize(arr)
    arr2 = np.array([], dtype=np.int64())
    assert check_tokenize(arr) != check_tokenize(arr2)