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
def test_normalize_base():
    for i in [1, 1.1, '1', slice(1, 2, 3), decimal.Decimal('1.1'), datetime.date(2021, 6, 25), pathlib.PurePath('/this/that')]:
        assert normalize_token(i) is i