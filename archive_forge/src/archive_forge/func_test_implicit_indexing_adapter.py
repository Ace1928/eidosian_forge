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
def test_implicit_indexing_adapter() -> None:
    array = np.arange(10, dtype=np.int64)
    implicit = indexing.ImplicitToExplicitIndexingAdapter(indexing.NumpyIndexingAdapter(array), indexing.BasicIndexer)
    np.testing.assert_array_equal(array, np.asarray(implicit))
    np.testing.assert_array_equal(array, implicit[:])