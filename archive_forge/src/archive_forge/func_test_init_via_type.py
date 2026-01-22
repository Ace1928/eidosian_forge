from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_init_via_type(self):
    a = xr.DataArray(np.random.rand(3, 4, 10), dims=['x', 'y', 'time'], coords={'area': (['x', 'y'], np.random.rand(3, 4))}).to_dataset(name='data')
    dt: DataTree = DataTree(data=a)

    def weighted_mean(ds):
        return ds.weighted(ds.area).mean(['x', 'y'])
    weighted_mean(dt.ds)