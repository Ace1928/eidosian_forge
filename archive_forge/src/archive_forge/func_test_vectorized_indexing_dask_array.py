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
@requires_dask
def test_vectorized_indexing_dask_array():
    darr = DataArray(data=[0.2, 0.4, 0.6], coords={'z': range(3)}, dims=('z',))
    indexer = DataArray(data=np.random.randint(0, 3, 8).reshape(4, 2).astype(int), coords={'y': range(4), 'x': range(2)}, dims=('y', 'x'))
    with pytest.raises(ValueError, match='Vectorized indexing with Dask arrays'):
        darr[indexer.chunk({'y': 2})]