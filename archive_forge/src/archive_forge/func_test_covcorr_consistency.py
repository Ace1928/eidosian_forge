from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
@pytest.mark.parametrize('n', range(9))
@pytest.mark.parametrize('dim', [None, 'time', 'x'])
def test_covcorr_consistency(n: int, dim: str | None, array_tuples: tuple[xr.DataArray, xr.DataArray]) -> None:
    da_a, da_b = array_tuples[n]
    da_a, da_b = broadcast(da_a, da_b)
    valid_values = da_a.notnull() & da_b.notnull()
    da_a = da_a.where(valid_values)
    da_b = da_b.where(valid_values)
    expected = xr.cov(da_a, da_b, dim=dim, ddof=0) / (da_a.std(dim=dim) * da_b.std(dim=dim))
    actual = xr.corr(da_a, da_b, dim=dim)
    assert_allclose(actual, expected)