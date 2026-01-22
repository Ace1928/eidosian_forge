from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_setitem_dataarray_replace_existing_node(self):
    t = xr.Dataset({'temp': [0, 50]})
    results: DataTree = DataTree(name='results', data=t)
    p = xr.DataArray(data=[2, 3])
    results['pressure'] = p
    expected = t.assign(pressure=p)
    xrt.assert_identical(results.to_dataset(), expected)