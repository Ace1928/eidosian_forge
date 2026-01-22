from __future__ import annotations
import itertools
from typing import Any
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable
from xarray.core import indexing, nputils
from xarray.core.indexes import PandasIndex, PandasMultiIndex
from xarray.core.types import T_Xarray
from xarray.tests import (
def test_indexing_1d_object_array() -> None:
    items = (np.arange(3), np.arange(6))
    arr = DataArray(np.array(items, dtype=object))
    actual = arr[0]
    expected_data = np.empty((), dtype=object)
    expected_data[()] = items[0]
    expected = DataArray(expected_data)
    assert [actual.data.item()] == [expected.data.item()]