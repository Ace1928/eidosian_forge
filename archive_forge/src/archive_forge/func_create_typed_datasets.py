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
def create_typed_datasets(num_datasets: int=2, seed: int | None=None) -> list[Dataset]:
    var_strings = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    result = []
    rng = np.random.default_rng(seed)
    lat = rng.standard_normal(size=(1, 4))
    lon = rng.standard_normal(size=(1, 4))
    for i in range(num_datasets):
        result.append(Dataset(data_vars={'float': (['x', 'y', 'day'], rng.standard_normal(size=(1, 4, 2))), 'float2': (['x', 'y', 'day'], rng.standard_normal(size=(1, 4, 2))), 'string': (['x', 'y', 'day'], rng.choice(var_strings, size=(1, 4, 2))), 'int': (['x', 'y', 'day'], rng.integers(0, 10, size=(1, 4, 2))), 'datetime64': (['x', 'y', 'day'], np.arange(np.datetime64('2017-01-01'), np.datetime64('2017-01-09')).reshape(1, 4, 2)), 'timedelta64': (['x', 'y', 'day'], np.reshape([pd.Timedelta(days=i) for i in range(8)], [1, 4, 2]))}, coords={'lat': (['x', 'y'], lat), 'lon': (['x', 'y'], lon), 'day': ['day' + str(i * 2 + 1), 'day' + str(i * 2 + 2)]}))
    return result