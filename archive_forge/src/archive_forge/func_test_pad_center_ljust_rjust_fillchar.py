from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_pad_center_ljust_rjust_fillchar(dtype) -> None:
    values = xr.DataArray(['a', 'bb', 'cccc', 'ddddd', 'eeeeee']).astype(dtype)
    result = values.str.center(5, fillchar='X')
    expected = xr.DataArray(['XXaXX', 'XXbbX', 'Xcccc', 'ddddd', 'eeeeee']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.pad(5, side='both', fillchar='X')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.ljust(5, fillchar='X')
    expected = xr.DataArray(['aXXXX', 'bbXXX', 'ccccX', 'ddddd', 'eeeeee']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected.astype(dtype))
    result = values.str.pad(5, side='right', fillchar='X')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.rjust(5, fillchar='X')
    expected = xr.DataArray(['XXXXa', 'XXXbb', 'Xcccc', 'ddddd', 'eeeeee']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected.astype(dtype))
    result = values.str.pad(5, side='left', fillchar='X')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    template = 'fillchar must be a character, not {dtype}'
    with pytest.raises(TypeError, match=template.format(dtype='str')):
        values.str.center(5, fillchar='XY')
    with pytest.raises(TypeError, match=template.format(dtype='str')):
        values.str.ljust(5, fillchar='XY')
    with pytest.raises(TypeError, match=template.format(dtype='str')):
        values.str.rjust(5, fillchar='XY')
    with pytest.raises(TypeError, match=template.format(dtype='str')):
        values.str.pad(5, fillchar='XY')