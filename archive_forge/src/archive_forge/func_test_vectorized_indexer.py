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
def test_vectorized_indexer() -> None:
    with pytest.raises(TypeError):
        check_integer(indexing.VectorizedIndexer)
    check_slice(indexing.VectorizedIndexer)
    check_array1d(indexing.VectorizedIndexer)
    check_array2d(indexing.VectorizedIndexer)
    with pytest.raises(ValueError, match='numbers of dimensions'):
        indexing.VectorizedIndexer((np.array(1, dtype=np.int64), np.arange(5, dtype=np.int64)))