from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_assign_when_already_child_with_variables_name(self):
    dt: DataTree = DataTree(data=None)
    DataTree(name='a', data=None, parent=dt)
    with pytest.raises(KeyError, match='names would collide'):
        dt.ds = xr.Dataset({'a': 0})
    dt.ds = xr.Dataset()
    new_ds = dt.to_dataset().assign(a=xr.DataArray(0))
    with pytest.raises(KeyError, match='names would collide'):
        dt.ds = new_ds