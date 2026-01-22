from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_create_two_children(self):
    root_data = xr.Dataset({'a': ('y', [6, 7, 8]), 'set0': ('x', [9, 10])})
    set1_data = xr.Dataset({'a': 0, 'b': 1})
    root: DataTree = DataTree(data=root_data)
    set1: DataTree = DataTree(name='set1', parent=root, data=set1_data)
    DataTree(name='set1', parent=root)
    DataTree(name='set2', parent=set1)