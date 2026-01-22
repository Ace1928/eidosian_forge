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
def test_tokenize_dict():
    assert check_tokenize({'x': 1, 1: 'x'}) == check_tokenize({1: 'x', 'x': 1})
    assert check_tokenize({'x': 1, 1: 'x'}) != check_tokenize({'x': 1, 2: 'x'})
    assert check_tokenize({'x': 1, 1: 'x'}) != check_tokenize({'x': 2, 1: 'x'})