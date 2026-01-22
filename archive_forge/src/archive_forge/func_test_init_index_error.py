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
def test_init_index_error(self) -> None:
    idx = PandasIndex([1, 2, 3], 'x')
    with pytest.raises(ValueError, match='no coordinate variables found'):
        Coordinates(indexes={'x': idx})
    with pytest.raises(TypeError, match='.* is not an `xarray.indexes.Index`'):
        Coordinates(coords={'x': ('x', [1, 2, 3])}, indexes={'x': 'not_an_xarray_index'})