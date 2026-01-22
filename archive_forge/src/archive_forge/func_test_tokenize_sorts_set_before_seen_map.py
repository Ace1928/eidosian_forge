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
def test_tokenize_sorts_set_before_seen_map():
    """Same as test_tokenize_sorts_dict_before_seen_map, but for sets.

    Note that this test is only meaningful if set insertion order impacts iteration
    order, which is an implementation detail of the Python interpreter.
    """
    v = (1, 2, 3)
    s1 = {(i, v) for i in range(100)}
    s2 = {(i, v) for i in reversed(range(100))}
    assert '__seen' in str(normalize_token(s1))
    assert check_tokenize(s1) == check_tokenize(s2)