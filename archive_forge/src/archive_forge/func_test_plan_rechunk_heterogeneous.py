from __future__ import annotations
import warnings
from itertools import product
import pytest
import math
import dask
import dask.array as da
from dask.array.rechunk import (
from dask.array.utils import assert_eq
from dask.utils import funcname
def test_plan_rechunk_heterogeneous():
    c = (10,) * 1
    f = (1,) * 10
    cf = c + f
    cc = c + c
    ff = f + f
    fc = f + c
    steps = _plan((cc, cf), (ff, ff))
    _assert_steps(steps, [(ff, ff)])
    steps = _plan((cf, fc), (ff, cf))
    _assert_steps(steps, [(ff, cf)])
    steps = _plan((cc, cf), (ff, cc))
    _assert_steps(steps, [(cc, cc), (ff, cc)])
    steps = _plan((cc, cf, cc), (ff, cc, cf))
    _assert_steps(steps, [(cc, cc, cc), (ff, cc, cf)])
    steps = _plan((cc, ff, cf), (ff, cf, cc), block_size_limit=100)
    _assert_steps(steps, [(cc, ff, cc), (ff, cf, cc)])