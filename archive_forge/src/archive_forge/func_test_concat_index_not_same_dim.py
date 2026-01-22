from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import PandasIndex
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
def test_concat_index_not_same_dim() -> None:
    ds1 = Dataset(coords={'x': ('x', [1, 2])})
    ds2 = Dataset(coords={'x': ('y', [3, 4])})
    ds2._indexes['x'] = PandasIndex([3, 4], 'y')
    with pytest.raises(ValueError, match="Cannot concatenate along dimension 'x' indexes with dimensions.*"):
        concat([ds1, ds2], dim='x')