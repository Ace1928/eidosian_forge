from __future__ import annotations
from numbers import Number
import numpy as np
import pytest
import xarray as xr
from xarray.backends.api import _get_default_engine
from xarray.tests import (
@pytest.mark.parametrize('shape,pref_chunks', [((5,), (2,)), ((5,), ((2, 2, 1),)), ((5, 6), (4, 2)), ((5, 6), (4, (2, 2, 2)))])
@pytest.mark.parametrize('request_with_empty_map', [False, True])
def test_honor_chunks(self, shape, pref_chunks, request_with_empty_map):
    """Honor the backend's preferred chunks when opening a dataset."""
    initial = self.create_dataset(shape, pref_chunks)
    chunks = {} if request_with_empty_map else dict.fromkeys(initial[self.var_name].dims, None)
    final = xr.open_dataset(initial, engine=PassThroughBackendEntrypoint, chunks=chunks)
    self.check_dataset(initial, final, explicit_chunks(pref_chunks, shape))