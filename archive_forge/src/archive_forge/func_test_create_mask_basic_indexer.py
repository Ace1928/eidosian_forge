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
def test_create_mask_basic_indexer() -> None:
    indexer = indexing.BasicIndexer((-1,))
    actual = indexing.create_mask(indexer, (3,))
    np.testing.assert_array_equal(True, actual)
    indexer = indexing.BasicIndexer((0,))
    actual = indexing.create_mask(indexer, (3,))
    np.testing.assert_array_equal(False, actual)