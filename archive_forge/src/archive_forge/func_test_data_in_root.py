from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_data_in_root(self):
    dat = xr.Dataset()
    dt = DataTree.from_dict({'/': dat})
    assert dt.name is None
    assert dt.parent is None
    assert dt.children == {}
    xrt.assert_identical(dt.to_dataset(), dat)