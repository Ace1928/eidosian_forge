from __future__ import annotations
from contextlib import suppress
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding import variables
from xarray.conventions import decode_cf_variable, encode_cf_variable
from xarray.tests import assert_allclose, assert_equal, assert_identical, requires_dask
@pytest.mark.parametrize('scale_factor', (10, [10]))
@pytest.mark.parametrize('add_offset', (0.1, [0.1]))
def test_scaling_offset_as_list(scale_factor, add_offset) -> None:
    encoding = dict(scale_factor=scale_factor, add_offset=add_offset)
    original = xr.Variable(('x',), np.arange(10.0), encoding=encoding)
    coder = variables.CFScaleOffsetCoder()
    encoded = coder.encode(original)
    roundtripped = coder.decode(encoded)
    assert_allclose(original, roundtripped)