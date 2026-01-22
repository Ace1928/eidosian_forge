from __future__ import annotations
import pytest
import xarray
from xarray import concat, merge
from xarray.backends.file_manager import FILE_CACHE
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.tests.test_dataset import create_test_data
def test_dataarray_attr_retention(self) -> None:
    da = create_test_dataarray_attrs()
    original_attrs = da.attrs
    result = da.mean()
    assert result.attrs == {}
    with xarray.set_options(keep_attrs='default'):
        result = da.mean()
        assert result.attrs == {}
    with xarray.set_options(keep_attrs=True):
        result = da.mean()
        assert result.attrs == original_attrs
    with xarray.set_options(keep_attrs=False):
        result = da.mean()
        assert result.attrs == {}