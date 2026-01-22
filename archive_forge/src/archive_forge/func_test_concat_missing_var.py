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
def test_concat_missing_var() -> None:
    datasets = create_concat_datasets(2, seed=123)
    expected = concat(datasets, dim='day')
    vars_to_drop = ['humidity', 'precipitation', 'cloud_cover']
    expected = expected.drop_vars(vars_to_drop)
    expected['pressure'][..., 2:] = np.nan
    datasets[0] = datasets[0].drop_vars(vars_to_drop)
    datasets[1] = datasets[1].drop_vars(vars_to_drop + ['pressure'])
    actual = concat(datasets, dim='day')
    assert list(actual.data_vars.keys()) == ['temperature', 'pressure']
    assert_identical(actual, expected)