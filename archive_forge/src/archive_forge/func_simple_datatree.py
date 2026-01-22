from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.core.datatree import DataTree
from xarray.tests import create_test_data, requires_dask
@pytest.fixture(scope='module')
def simple_datatree(create_test_datatree):
    """
    Invoke create_test_datatree fixture (callback).

    Returns a DataTree.
    """
    return create_test_datatree()