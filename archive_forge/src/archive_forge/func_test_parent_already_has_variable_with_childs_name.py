from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_parent_already_has_variable_with_childs_name(self):
    dt: DataTree = DataTree(data=xr.Dataset({'a': [0], 'b': 1}))
    with pytest.raises(KeyError, match='already contains a data variable named a'):
        DataTree(name='a', data=None, parent=dt)