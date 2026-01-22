from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_path_property(self):
    sue: DataTree = DataTree()
    mary: DataTree = DataTree(children={'Sue': sue})
    john: DataTree = DataTree(children={'Mary': mary})
    assert sue.path == '/Mary/Sue'
    assert john.path == '/'