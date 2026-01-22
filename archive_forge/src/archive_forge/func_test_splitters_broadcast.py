from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_splitters_broadcast(dtype) -> None:
    values = xr.DataArray(['ab cd,de fg', 'spam, ,eggs swallow', 'red_blue'], dims=['X']).astype(dtype)
    sep = xr.DataArray([' ', ','], dims=['Y']).astype(dtype)
    expected_left = xr.DataArray([[['ab', 'cd,de fg'], ['ab cd', 'de fg']], [['spam,', ',eggs swallow'], ['spam', ' ,eggs swallow']], [['red_blue', ''], ['red_blue', '']]], dims=['X', 'Y', 'ZZ']).astype(dtype)
    expected_right = xr.DataArray([[['ab cd,de', 'fg'], ['ab cd', 'de fg']], [['spam, ,eggs', 'swallow'], ['spam, ', 'eggs swallow']], [['', 'red_blue'], ['', 'red_blue']]], dims=['X', 'Y', 'ZZ']).astype(dtype)
    res_left = values.str.split(dim='ZZ', sep=sep, maxsplit=1)
    res_right = values.str.rsplit(dim='ZZ', sep=sep, maxsplit=1)
    assert_equal(res_left, expected_left)
    assert_equal(res_right, expected_right)
    expected_left = xr.DataArray([[['ab', ' ', 'cd,de fg'], ['ab cd', ',', 'de fg']], [['spam,', ' ', ',eggs swallow'], ['spam', ',', ' ,eggs swallow']], [['red_blue', '', ''], ['red_blue', '', '']]], dims=['X', 'Y', 'ZZ']).astype(dtype)
    expected_right = xr.DataArray([[['ab', ' ', 'cd,de fg'], ['ab cd', ',', 'de fg']], [['spam,', ' ', ',eggs swallow'], ['spam', ',', ' ,eggs swallow']], [['red_blue', '', ''], ['red_blue', '', '']]], dims=['X', 'Y', 'ZZ']).astype(dtype)
    res_left = values.str.partition(dim='ZZ', sep=sep)
    res_right = values.str.partition(dim='ZZ', sep=sep)
    assert_equal(res_left, expected_left)
    assert_equal(res_right, expected_right)