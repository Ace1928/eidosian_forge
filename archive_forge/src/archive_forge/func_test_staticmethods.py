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
def test_staticmethods():
    a, b, c = (HasStaticMethods(1), HasStaticMethods(2), HasStaticMethods2(1))
    assert check_tokenize(a.normal_method) != check_tokenize(b.normal_method)
    assert check_tokenize(a.normal_method) != check_tokenize(c.normal_method)
    assert check_tokenize(a.static_method) == check_tokenize(b.static_method)
    assert check_tokenize(a.static_method) == check_tokenize(c.static_method)
    assert check_tokenize(a.class_method) == check_tokenize(b.class_method)
    assert check_tokenize(a.class_method) != check_tokenize(c.class_method)