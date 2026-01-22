from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
@pytest.mark.xfail(reason="assigning Datasets doesn't yet create new nodes")
def test_setitem_dataset_as_new_node(self):
    data = xr.Dataset({'temp': [0, 50]})
    folder1: DataTree = DataTree(name='folder1')
    folder1['results'] = data
    xrt.assert_identical(folder1['results'].to_dataset(), data)