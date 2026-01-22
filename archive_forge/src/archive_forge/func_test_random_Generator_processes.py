from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask import config
from dask.array.utils import assert_eq
@pytest.mark.parametrize('backend', ['cupy', 'numpy'])
def test_random_Generator_processes(backend):
    with config.set({'array.backend': backend}):
        state = da.random.default_rng(5)
        x = state.standard_normal(size=(2, 3), chunks=3)
        state = da.random.default_rng(5)
        y = state.standard_normal(size=(2, 3), chunks=3)
        assert_eq(x, y, scheduler='processes')