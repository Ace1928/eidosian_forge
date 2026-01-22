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
def test_index_scalar(self) -> None:
    x = indexing.MemoryCachedArray(np.array(['foo', 'bar']))
    assert np.array(x[B[0]][B[()]]) == 'foo'