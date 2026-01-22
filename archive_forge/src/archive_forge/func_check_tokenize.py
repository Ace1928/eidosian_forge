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
def check_tokenize(*args, **kwargs):
    with dask.config.set({'tokenize.ensure-deterministic': True}):
        before = tokenize(*args, **kwargs)
        after = tokenize(*args, **kwargs)
        assert before == after
        args2, kwargs2 = cloudpickle.loads(cloudpickle.dumps((args, kwargs)))
        args3, kwargs3 = cloudpickle.loads(cloudpickle.dumps((args, kwargs)))
        args3, kwargs3 = cloudpickle.loads(cloudpickle.dumps((args3, kwargs3)))
        tok2 = tokenize(*args2, **kwargs2)
        tok3 = tokenize(*args3, **kwargs3)
        assert tok2 == tok3
    return before