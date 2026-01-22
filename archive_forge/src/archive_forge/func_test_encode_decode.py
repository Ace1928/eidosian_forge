from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_encode_decode() -> None:
    data = xr.DataArray(['a', 'b', 'aä'])
    encoded = data.str.encode('utf-8')
    decoded = encoded.str.decode('utf-8')
    assert data.dtype == decoded.dtype
    assert_equal(data, decoded)