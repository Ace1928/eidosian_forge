from __future__ import annotations
import pytest
import xarray
from xarray import concat, merge
from xarray.backends.file_manager import FILE_CACHE
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.tests.test_dataset import create_test_data
def test_invalid_option_raises() -> None:
    with pytest.raises(ValueError):
        xarray.set_options(not_a_valid_options=True)