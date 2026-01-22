from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_ipython_key_completions(self, create_test_datatree):
    dt = create_test_datatree()
    key_completions = dt._ipython_key_completions_()
    node_keys = [node.path[1:] for node in dt.subtree]
    assert all((node_key in key_completions for node_key in node_keys))
    var_keys = list(dt.variables.keys())
    assert all((var_key in key_completions for var_key in var_keys))