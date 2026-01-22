from __future__ import annotations
from contextlib import suppress
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding import variables
from xarray.conventions import decode_cf_variable, encode_cf_variable
from xarray.tests import assert_allclose, assert_equal, assert_identical, requires_dask
def test_CFMaskCoder_missing_value() -> None:
    expected = xr.DataArray(np.array([[26915, 27755, -9999, 27705], [25595, -9999, 28315, -9999]]), dims=['npts', 'ntimes'], name='tmpk')
    expected.attrs['missing_value'] = -9999
    decoded = xr.decode_cf(expected.to_dataset())
    encoded, _ = xr.conventions.cf_encoder(decoded.variables, decoded.attrs)
    assert_equal(encoded['tmpk'], expected.variable)
    decoded.tmpk.encoding['_FillValue'] = -9940
    with pytest.raises(ValueError):
        encoded, _ = xr.conventions.cf_encoder(decoded.variables, decoded.attrs)