from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from xarray.core.alignment import align
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import PandasIndex, PandasMultiIndex
from xarray.core.variable import IndexVariable, Variable
from xarray.tests import assert_identical, source_ndarray
def test_dataset_from_coords_with_multidim_var_same_name(self):
    var = Variable(data=np.arange(6).reshape(2, 3), dims=['x', 'y'])
    coords = Coordinates(coords={'x': var}, indexes={})
    ds = Dataset(coords=coords)
    assert ds.coords['x'].dims == ('x', 'y')