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
@pytest.mark.parametrize('other', [(2002, 6, 25), (2021, 7, 25), (2021, 6, 26)])
def test_tokenize_datetime_date(other):
    a = datetime.date(2021, 6, 25)
    b = datetime.date(*other)
    assert check_tokenize(a) != check_tokenize(b)