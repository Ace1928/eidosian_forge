from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_is_hollow(self):
    john: DataTree = DataTree(data=xr.Dataset({'a': 0}))
    assert john.is_hollow
    eve: DataTree = DataTree(children={'john': john})
    assert eve.is_hollow
    eve.ds = xr.Dataset({'a': 1})
    assert not eve.is_hollow