from __future__ import annotations
from itertools import combinations, permutations
from typing import cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import _parse_array_of_cftime_strings
from xarray.core.types import InterpOptions
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
def get_example_data(case: int) -> xr.DataArray:
    if case == 0:
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 0.1, 30)
        return xr.DataArray(np.sin(x[:, np.newaxis]) * np.cos(y), dims=['x', 'y'], coords={'x': x, 'y': y, 'x2': ('x', x ** 2)})
    elif case == 1:
        return get_example_data(0).chunk({'y': 3})
    elif case == 2:
        return get_example_data(0).chunk({'x': 25, 'y': 3})
    elif case == 3:
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 0.1, 30)
        z = np.linspace(0.1, 0.2, 10)
        return xr.DataArray(np.sin(x[:, np.newaxis, np.newaxis]) * np.cos(y[:, np.newaxis]) * z, dims=['x', 'y', 'z'], coords={'x': x, 'y': y, 'x2': ('x', x ** 2), 'z': z})
    elif case == 4:
        return get_example_data(3).chunk({'z': 5})
    else:
        raise ValueError('case must be 1-4')