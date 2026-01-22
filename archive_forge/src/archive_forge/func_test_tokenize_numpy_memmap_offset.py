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
@pytest.mark.skipif('not np')
def test_tokenize_numpy_memmap_offset(tmpdir):
    fn = str(tmpdir.join('demo_data'))
    with open(fn, 'wb') as f:
        f.write(b'ashekwicht')
    with open(fn, 'rb') as f:
        mmap1 = np.memmap(f, dtype=np.uint8, mode='r', offset=0, shape=5)
        mmap2 = np.memmap(f, dtype=np.uint8, mode='r', offset=5, shape=5)
        mmap3 = np.memmap(f, dtype=np.uint8, mode='r', offset=0, shape=5)
        assert check_tokenize(mmap1) == check_tokenize(mmap1)
        assert check_tokenize(mmap1) == check_tokenize(mmap3)
        assert check_tokenize(mmap1) != check_tokenize(mmap2)
        assert check_tokenize(mmap1[1:-1]) == check_tokenize(mmap1[1:-1])
        assert check_tokenize(mmap1[1:-1]) == check_tokenize(mmap3[1:-1])
        assert check_tokenize(mmap1[1:2]) == check_tokenize(mmap3[1:2])
        assert check_tokenize(mmap1[1:2]) != check_tokenize(mmap1[1:3])
        assert check_tokenize(mmap1[1:2]) != check_tokenize(mmap3[1:3])
        sub1 = mmap1[1:-1]
        sub2 = mmap2[1:-1]
        assert check_tokenize(sub1) != check_tokenize(sub2)