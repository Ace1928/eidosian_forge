from __future__ import annotations
import copy
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.indexes import (
from xarray.core.variable import IndexVariable, Variable
from xarray.tests import assert_array_equal, assert_identical, requires_cftime
from xarray.tests.test_coding_times import _all_cftime_date_types
@pytest.fixture
def indexes_and_vars(self) -> tuple[list[PandasIndex], dict[Hashable, Variable]]:
    x_idx = PandasIndex(pd.Index([1, 2, 3], name='x'), 'x')
    y_idx = PandasIndex(pd.Index([4, 5, 6], name='y'), 'y')
    z_pd_midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=['one', 'two'])
    z_midx = PandasMultiIndex(z_pd_midx, 'z')
    indexes = [x_idx, y_idx, z_midx]
    variables = {}
    for idx in indexes:
        variables.update(idx.create_variables())
    return (indexes, variables)