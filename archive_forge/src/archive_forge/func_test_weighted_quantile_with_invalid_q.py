from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize('q', (-1, 1.1, (0.5, 1.1), ((0.2, 0.4), (0.6, 0.8))))
def test_weighted_quantile_with_invalid_q(q):
    da = DataArray([1, 1.9, 2.2, 3, 3.7, 4.1, 5])
    q = np.asarray(q)
    weights = xr.ones_like(da)
    if q.ndim <= 1:
        with pytest.raises(ValueError, match='q values must be between 0 and 1'):
            da.weighted(weights).quantile(q)
    else:
        with pytest.raises(ValueError, match='q must be a scalar or 1d'):
            da.weighted(weights).quantile(q)