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
def test_plan_rechunk():
    c = (20,) * 2
    f = (2,) * 20
    nc = (np.nan,) * 2
    nf = (np.nan,) * 20
    steps = _plan((), ())
    _assert_steps(steps, [()])
    steps = _plan((c, ()), (f, ()))
    _assert_steps(steps, [(f, ())])
    steps = _plan((c,), (f,))
    _assert_steps(steps, [(f,)])
    steps = _plan((f,), (c,))
    _assert_steps(steps, [(c,)])
    steps = _plan((c, c), (f, f))
    _assert_steps(steps, [(f, f)])
    steps = _plan((f, f), (c, c))
    _assert_steps(steps, [(c, c)])
    steps = _plan((f, c), (c, c))
    _assert_steps(steps, [(c, c)])
    steps = _plan((c, c, c, c), (c, f, c, c))
    _assert_steps(steps, [(c, f, c, c)])
    steps = _plan((f, c), (c, f))
    _assert_steps(steps, [(c, c), (c, f)])
    steps = _plan((c + c, c + f), (f + f, c + c))
    _assert_steps(steps, [(c + c, c + c), (f + f, c + c)])
    steps = _plan((nc + nf, c + c, c + f), (nc + nf, f + f, c + c))
    _assert_steps(steps, steps)
    steps = _plan((c, c), (f, f), threshold=1)
    _assert_steps(steps, [(f, f)])
    steps = _plan((f, c), (c, f), block_size_limit=400)
    _assert_steps(steps, [(c, c), (c, f)])
    m = (10,) * 4
    steps = _plan((f, c), (c, f), block_size_limit=399)
    _assert_steps(steps, [(m, c), (c, f)])
    steps2 = _plan((f, c), (c, f), block_size_limit=3999, itemsize=10)
    _assert_steps(steps2, steps)
    c = (1000,) * 2
    f = (2,) * 1000
    steps = _plan((f, c), (c, f), block_size_limit=99999)
    assert len(steps) == 3
    assert steps[-1] == (c, f)
    for i in range(len(steps) - 1):
        prev = steps[i]
        succ = steps[i + 1]
        assert len(succ[0]) <= len(prev[0]) / 2.0
        assert len(succ[1]) >= len(prev[1]) * 2.0