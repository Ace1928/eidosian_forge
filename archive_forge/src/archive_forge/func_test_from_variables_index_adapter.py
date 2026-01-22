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
def test_from_variables_index_adapter(self) -> None:
    data = pd.Series(['foo', 'bar'], dtype='category')
    pd_idx = pd.Index(data)
    var = xr.Variable('x', pd_idx)
    index = PandasIndex.from_variables({'x': var}, options={})
    assert isinstance(index.index, pd.CategoricalIndex)