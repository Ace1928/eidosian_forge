from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_grafted_subtree_retains_name(self):
    subtree: DataTree = DataTree(name='original_subtree_name')
    root: DataTree = DataTree(name='root')
    root['new_subtree_name'] = subtree
    assert subtree.name == 'original_subtree_name'