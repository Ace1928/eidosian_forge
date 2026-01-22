from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_starts_ends_with_broadcast(dtype) -> None:
    values = xr.DataArray(['om', 'foo_nom', 'nom', 'bar_foo', 'foo_bar'], dims='X').astype(dtype)
    pat = xr.DataArray(['foo', 'bar'], dims='Y').astype(dtype)
    result = values.str.startswith(pat)
    expected = xr.DataArray([[False, False], [True, False], [False, False], [False, True], [True, False]], dims=['X', 'Y'])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.endswith(pat)
    expected = xr.DataArray([[False, False], [False, False], [False, False], [True, False], [False, True]], dims=['X', 'Y'])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)