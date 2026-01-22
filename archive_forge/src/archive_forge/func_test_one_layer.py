from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_one_layer(self):
    dat1, dat2 = (xr.Dataset({'a': 1}), xr.Dataset({'b': 2}))
    dt = DataTree.from_dict({'run1': dat1, 'run2': dat2})
    xrt.assert_identical(dt.to_dataset(), xr.Dataset())
    assert dt.name is None
    xrt.assert_identical(dt['run1'].to_dataset(), dat1)
    assert dt['run1'].children == {}
    xrt.assert_identical(dt['run2'].to_dataset(), dat2)
    assert dt['run2'].children == {}