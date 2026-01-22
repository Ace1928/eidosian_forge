from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_setitem_overwrite_data_in_node_with_none(self):
    john: DataTree = DataTree(name='john')
    mary: DataTree = DataTree(name='mary', parent=john, data=xr.Dataset())
    john['mary'] = DataTree()
    xrt.assert_identical(mary.to_dataset(), xr.Dataset())
    john.ds = xr.Dataset()
    with pytest.raises(ValueError, match='has no name'):
        john['.'] = DataTree()