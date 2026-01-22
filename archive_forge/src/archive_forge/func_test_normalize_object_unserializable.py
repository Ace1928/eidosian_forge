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
def test_normalize_object_unserializable():

    class C:

        def __reduce__(self):
            assert False
    c = C()
    with dask.config.set({'tokenize.ensure-deterministic': False}):
        assert tokenize(c) != tokenize(c)
        assert normalize_token(c) != normalize_token(c)
    with dask.config.set({'tokenize.ensure-deterministic': True}):
        with pytest.raises(TokenizationError, match='cannot be deterministically hashed'):
            tokenize(c)
    assert tokenize(c, ensure_deterministic=False) != tokenize(c, ensure_deterministic=False)
    with pytest.raises(TokenizationError, match='cannot be deterministically hashed'):
        tokenize(c, ensure_deterministic=True)