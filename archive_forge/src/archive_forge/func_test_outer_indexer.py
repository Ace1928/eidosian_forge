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
def test_outer_indexer() -> None:
    check_integer(indexing.OuterIndexer)
    check_slice(indexing.OuterIndexer)
    check_array1d(indexing.OuterIndexer)
    with pytest.raises(TypeError):
        check_array2d(indexing.OuterIndexer)