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
def test_init_from_coords(self) -> None:
    expected = Dataset(coords={'foo': ('x', [0, 1, 2])})
    coords = Coordinates(coords=expected.coords)
    assert_identical(coords.to_dataset(), expected)
    assert coords.variables['foo'] is not expected.variables['foo']
    expected = Dataset(coords={'x': [0, 1, 2]})
    coords = Coordinates(coords=expected.coords)
    assert_identical(coords.to_dataset(), expected)
    assert expected.xindexes == coords.xindexes
    with pytest.raises(ValueError, match='passing both.*Coordinates.*indexes.*not allowed'):
        coords = Coordinates(coords=expected.coords, indexes={'x': PandasIndex([0, 1, 2], 'x')})