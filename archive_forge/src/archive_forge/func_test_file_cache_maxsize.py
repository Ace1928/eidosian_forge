from __future__ import annotations
import pytest
import xarray
from xarray import concat, merge
from xarray.backends.file_manager import FILE_CACHE
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.tests.test_dataset import create_test_data
def test_file_cache_maxsize() -> None:
    with pytest.raises(ValueError):
        xarray.set_options(file_cache_maxsize=0)
    original_size = FILE_CACHE.maxsize
    with xarray.set_options(file_cache_maxsize=123):
        assert FILE_CACHE.maxsize == 123
    assert FILE_CACHE.maxsize == original_size