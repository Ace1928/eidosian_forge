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
def test_tokenize_partial_func_args_kwargs_consistent():
    f = partial(f3, f2, c=f1)
    g = partial(f3, f2, c=f1)
    h = partial(f3, f2, c=5)
    assert check_tokenize(f) == check_tokenize(g)
    assert check_tokenize(f) != check_tokenize(h)