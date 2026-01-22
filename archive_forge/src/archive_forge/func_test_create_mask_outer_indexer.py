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
def test_create_mask_outer_indexer() -> None:
    indexer = indexing.OuterIndexer((np.array([0, -1, 2]),))
    expected = np.array([False, True, False])
    actual = indexing.create_mask(indexer, (5,))
    np.testing.assert_array_equal(expected, actual)
    indexer = indexing.OuterIndexer((1, slice(2), np.array([0, -1, 2])))
    expected = np.array(2 * [[False, True, False]])
    actual = indexing.create_mask(indexer, (5, 5, 5))
    np.testing.assert_array_equal(expected, actual)