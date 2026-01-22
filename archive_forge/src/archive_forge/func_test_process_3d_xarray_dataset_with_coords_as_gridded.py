import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_process_3d_xarray_dataset_with_coords_as_gridded(self):
    import xarray as xr
    kwargs = self.default_kwargs
    kwargs.update(gridded=True, x='lon', y='lat')
    data, x, y, by, groupby = process_xarray(data=self.ds, **kwargs)
    assert isinstance(data, xr.Dataset)
    assert list(data.data_vars.keys()) == ['air']
    assert x == 'lon'
    assert y == 'lat'
    assert by is None
    assert groupby == ['time']