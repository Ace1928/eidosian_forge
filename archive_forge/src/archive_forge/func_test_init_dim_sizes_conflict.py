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
def test_init_dim_sizes_conflict(self) -> None:
    with pytest.raises(ValueError):
        Coordinates(coords={'foo': ('x', [1, 2]), 'bar': ('x', [1, 2, 3, 4])})