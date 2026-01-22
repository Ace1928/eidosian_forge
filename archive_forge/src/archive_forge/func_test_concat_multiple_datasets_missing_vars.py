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
@pytest.mark.parametrize('include_day', [True, False])
def test_concat_multiple_datasets_missing_vars(include_day: bool) -> None:
    vars_to_drop = ['temperature', 'pressure', 'humidity', 'precipitation', 'cloud_cover']
    datasets = create_concat_datasets(len(vars_to_drop), seed=123, include_day=include_day)
    expected = concat(datasets, dim='day')
    for i, name in enumerate(vars_to_drop):
        if include_day:
            expected[name][..., i * 2:(i + 1) * 2] = np.nan
        else:
            expected[name][i:i + 1, ...] = np.nan
    datasets = [ds.drop_vars(varname) for ds, varname in zip(datasets, vars_to_drop)]
    actual = concat(datasets, dim='day')
    assert list(actual.data_vars.keys()) == ['pressure', 'humidity', 'precipitation', 'cloud_cover', 'temperature']
    assert_identical(actual, expected)