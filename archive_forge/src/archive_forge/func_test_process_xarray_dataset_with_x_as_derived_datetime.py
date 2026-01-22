import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_process_xarray_dataset_with_x_as_derived_datetime(self):
    import pandas as pd
    data = self.ds.mean(dim=['lat', 'lon'])
    kwargs = self.default_kwargs
    kwargs.update(gridded=False, y='air', x='time.dayofyear')
    data, x, y, by, groupby = process_xarray(data=data, **kwargs)
    assert isinstance(data, pd.DataFrame)
    assert x == 'time.dayofyear'
    assert y == 'air'
    assert not by
    assert not groupby