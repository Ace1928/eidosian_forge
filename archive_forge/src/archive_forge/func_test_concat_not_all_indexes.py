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
def test_concat_not_all_indexes() -> None:
    ds1 = Dataset(coords={'x': ('x', [1, 2])})
    ds2 = Dataset(coords={'x': ('y', [3, 4])})
    with pytest.raises(ValueError, match="'x' must have either an index or no index in all datasets.*"):
        concat([ds1, ds2], dim='x')