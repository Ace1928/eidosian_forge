from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_two_layers(self):
    dat1, dat2 = (xr.Dataset({'a': 1}), xr.Dataset({'a': [1, 2]}))
    dt = DataTree.from_dict({'highres/run': dat1, 'lowres/run': dat2})
    assert 'highres' in dt.children
    assert 'lowres' in dt.children
    highres_run = dt['highres/run']
    xrt.assert_identical(highres_run.to_dataset(), dat1)