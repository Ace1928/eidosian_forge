from __future__ import annotations
from contextlib import suppress
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding import variables
from xarray.conventions import decode_cf_variable, encode_cf_variable
from xarray.tests import assert_allclose, assert_equal, assert_identical, requires_dask
@pytest.mark.parametrize('bits', [1, 2, 4, 8])
def test_decode_signed_from_unsigned(bits) -> None:
    unsigned_dtype = np.dtype(f'u{bits}')
    signed_dtype = np.dtype(f'i{bits}')
    original_values = np.array([-1], dtype=signed_dtype)
    encoded = xr.Variable(('x',), original_values.astype(unsigned_dtype), attrs={'_Unsigned': 'false'})
    coder = variables.UnsignedIntegerCoder()
    decoded = coder.decode(encoded)
    assert decoded.dtype == signed_dtype
    assert decoded.values == original_values