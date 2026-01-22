from __future__ import annotations
from numbers import Number
import numpy as np
import pytest
import xarray as xr
from xarray.backends.api import _get_default_engine
from xarray.tests import (
def test_multiindex() -> None:
    dataset = xr.Dataset(coords={'coord1': ['A', 'B'], 'coord2': [1, 2]})
    dataset = dataset.stack(z=['coord1', 'coord2'])

    class MultiindexBackend(xr.backends.BackendEntrypoint):

        def open_dataset(self, filename_or_obj, drop_variables=None, **kwargs) -> xr.Dataset:
            return dataset.copy(deep=True)
    loaded = xr.open_dataset('fake_filename', engine=MultiindexBackend)
    assert_identical(dataset, loaded)