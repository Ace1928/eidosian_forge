from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_immutability(self):
    dt: DataTree = DataTree(name='root', data=None)
    DataTree(name='a', data=None, parent=dt)
    with pytest.raises(AttributeError, match='Mutation of the DatasetView is not allowed'):
        dt.ds['a'] = xr.DataArray(0)
    with pytest.raises(AttributeError, match='Mutation of the DatasetView is not allowed'):
        dt.ds.update({'a': 0})