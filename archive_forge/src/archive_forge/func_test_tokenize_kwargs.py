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
def test_tokenize_kwargs():
    check_tokenize(5, x=1)
    assert check_tokenize(5) != check_tokenize(5, x=1)
    assert check_tokenize(5, x=1) != check_tokenize(5, x=2)
    assert check_tokenize(5, x=1) != check_tokenize(5, y=1)
    assert check_tokenize(5, foo='bar') != check_tokenize(5, {'foo': 'bar'})