from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_create_with_data(self):
    dat = xr.Dataset({'a': 0})
    john: DataTree = DataTree(name='john', data=dat)
    xrt.assert_identical(john.to_dataset(), dat)
    with pytest.raises(TypeError):
        DataTree(name='mary', parent=john, data='junk')