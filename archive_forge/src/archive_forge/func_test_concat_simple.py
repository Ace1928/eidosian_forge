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
@pytest.mark.parametrize('coords', ['different', 'minimal'])
@pytest.mark.parametrize('dim', ['dim1', 'dim2'])
def test_concat_simple(self, data, dim, coords) -> None:
    datasets = [g for _, g in data.groupby(dim, squeeze=False)]
    assert_identical(data, concat(datasets, dim, coords=coords))