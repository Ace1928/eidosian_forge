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
def test_tokenize_random_state_numpy():
    a = np.random.RandomState(123)
    b = np.random.RandomState(123)
    c = np.random.RandomState(456)
    assert check_tokenize(a) == check_tokenize(b)
    assert check_tokenize(a) != check_tokenize(c)
    a.random()
    assert check_tokenize(a) != check_tokenize(b)