from __future__ import annotations
import warnings
import pytest
from packaging.version import parse as parse_version
import numpy as np
import dask.array as da
import dask.array.stats
from dask.array.utils import allclose, assert_eq
from dask.delayed import Delayed
def test_power_divergence_invalid():
    a = np.random.random(size=30)
    a_ = da.from_array(a, 3)
    with pytest.raises(ValueError):
        dask.array.stats.power_divergence(a_, lambda_='wrong')