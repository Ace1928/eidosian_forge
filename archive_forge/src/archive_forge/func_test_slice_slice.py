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
def test_slice_slice(self) -> None:
    arr = ReturnItem()
    for size in [100, 99]:
        x = np.arange(size)
        slices = [arr[:3], arr[:4], arr[2:4], arr[:1], arr[:-1], arr[5:-1], arr[-5:-1], arr[::-1], arr[5::-1], arr[:3:-1], arr[:30:-1], arr[10:4], arr[::4], arr[4:4:4], arr[:4:-4], arr[::-2]]
        for i in slices:
            for j in slices:
                expected = x[i][j]
                new_slice = indexing.slice_slice(i, j, size=size)
                actual = x[new_slice]
                assert_array_equal(expected, actual)