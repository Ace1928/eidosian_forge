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
def test_stack_non_unique(self) -> None:
    prod_vars = {'x': xr.Variable('x', pd.Index(['b', 'a']), attrs={'foo': 'bar'}), 'y': xr.Variable('y', pd.Index([1, 1, 2]))}
    index = PandasMultiIndex.stack(prod_vars, 'z')
    np.testing.assert_array_equal(index.index.codes, [[0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 0, 1]])
    np.testing.assert_array_equal(index.index.levels[0], ['b', 'a'])
    np.testing.assert_array_equal(index.index.levels[1], [1, 2])