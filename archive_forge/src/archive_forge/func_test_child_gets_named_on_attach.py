from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_child_gets_named_on_attach(self):
    sue: DataTree = DataTree()
    mary: DataTree = DataTree(children={'Sue': sue})
    assert sue.name == 'Sue'