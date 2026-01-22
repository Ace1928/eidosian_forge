from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_named_self(self, create_test_datatree):
    dt = create_test_datatree()

    def f(x, tree, y):
        tree.attrs.update({'x': x, 'y': y})
        return tree
    attrs = {'x': 1, 'y': 2}
    actual = dt.pipe((f, 'tree'), **attrs)
    assert actual is dt and actual.attrs == attrs