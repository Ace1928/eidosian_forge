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
def nested_tokenize_ensure_deterministic():
    """Test that the ensure_deterministic override is not lost if tokenize() is
    called recursively
    """

    class C:

        def __dask_tokenize__(self):
            return tokenize(object())
    assert tokenize(C(), ensure_deterministic=False) != tokenize(C(), ensure_deterministic=False)
    with pytest.raises(TokenizationError):
        tokenize(C())