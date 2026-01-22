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
def test_nested_tokenize_seen():
    """Test that calling tokenize() recursively doesn't alter the output due to
    memoization of already-seen objects
    """
    o = [1, 2, 3]

    class C:

        def __init__(self, x):
            self.x = x
            self.tok = None

        def __dask_tokenize__(self):
            if not self.tok:
                self.tok = tokenize(self.x)
            return self.tok
    c1, c2 = (C(o), C(o))
    check_tokenize(o, c1, o)
    assert c1.tok
    assert check_tokenize(c1) == check_tokenize(c2)