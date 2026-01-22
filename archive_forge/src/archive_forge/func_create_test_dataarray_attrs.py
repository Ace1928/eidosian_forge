from __future__ import annotations
import pytest
import xarray
from xarray import concat, merge
from xarray.backends.file_manager import FILE_CACHE
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.tests.test_dataset import create_test_data
def create_test_dataarray_attrs(seed=0, var='var1'):
    da = create_test_data(seed)[var]
    da.attrs = {'attr1': 5, 'attr2': 'history', 'attr3': {'nested': 'more_info'}}
    return da