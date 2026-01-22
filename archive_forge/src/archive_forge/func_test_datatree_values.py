from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_datatree_values(self):
    dat1: DataTree = DataTree(data=xr.Dataset({'a': 1}))
    expected: DataTree = DataTree()
    expected['a'] = dat1
    actual = DataTree.from_dict({'a': dat1})
    dtt.assert_identical(actual, expected)