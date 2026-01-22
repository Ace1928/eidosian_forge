from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_replace_callable() -> None:
    values = xr.DataArray(['fooBAD__barBAD'])
    repl = lambda m: m.group(0).swapcase()
    result = values.str.replace('[a-z][A-Z]{2}', repl, n=2)
    exp = xr.DataArray(['foObaD__baRbaD'])
    assert result.dtype == exp.dtype
    assert_equal(result, exp)
    values = xr.DataArray(['Foo Bar Baz'])
    pat = '(?P<first>\\w+) (?P<middle>\\w+) (?P<last>\\w+)'
    repl = lambda m: m.group('middle').swapcase()
    result = values.str.replace(pat, repl)
    exp = xr.DataArray(['bAR'])
    assert result.dtype == exp.dtype
    assert_equal(result, exp)
    values = xr.DataArray(['Foo Bar Baz'], dims=['x'])
    pat = '(?P<first>\\w+) (?P<middle>\\w+) (?P<last>\\w+)'
    repl2 = xr.DataArray([lambda m: m.group('first').swapcase(), lambda m: m.group('middle').swapcase(), lambda m: m.group('last').swapcase()], dims=['Y'])
    result = values.str.replace(pat, repl2)
    exp = xr.DataArray([['fOO', 'bAR', 'bAZ']], dims=['x', 'Y'])
    assert result.dtype == exp.dtype
    assert_equal(result, exp)