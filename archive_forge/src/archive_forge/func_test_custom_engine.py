from __future__ import annotations
from numbers import Number
import numpy as np
import pytest
import xarray as xr
from xarray.backends.api import _get_default_engine
from xarray.tests import (
def test_custom_engine() -> None:
    expected = xr.Dataset(dict(a=2 * np.arange(5)), coords=dict(x=('x', np.arange(5), dict(units='s'))))

    class CustomBackend(xr.backends.BackendEntrypoint):

        def open_dataset(self, filename_or_obj, drop_variables=None, **kwargs) -> xr.Dataset:
            return expected.copy(deep=True)
    actual = xr.open_dataset('fake_filename', engine=CustomBackend)
    assert_identical(expected, actual)