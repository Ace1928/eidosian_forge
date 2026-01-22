from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_nones(self):
    dt = DataTree.from_dict({'d': None, 'd/e': None})
    assert [node.name for node in dt.subtree] == [None, 'd', 'e']
    assert [node.path for node in dt.subtree] == ['/', '/d', '/d/e']
    xrt.assert_identical(dt['d/e'].to_dataset(), xr.Dataset())