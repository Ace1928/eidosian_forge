from __future__ import annotations
import pytest
import xarray
from xarray import concat, merge
from xarray.backends.file_manager import FILE_CACHE
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.tests.test_dataset import create_test_data
def test_nested_options() -> None:
    original = OPTIONS['display_width']
    with xarray.set_options(display_width=1):
        assert OPTIONS['display_width'] == 1
        with xarray.set_options(display_width=2):
            assert OPTIONS['display_width'] == 2
        assert OPTIONS['display_width'] == 1
    assert OPTIONS['display_width'] == original