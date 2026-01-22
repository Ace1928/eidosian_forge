from __future__ import annotations
import warnings
from abc import ABC
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Generic
import numpy as np
import pandas as pd
import pytest
import pytz
from xarray import DataArray, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
from xarray.core.types import T_DuckArray
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_namedarray import NamedArraySubclassobjects
@requires_pandas_version_two
@pytest.mark.parametrize(('index', 'dtype'), [(pd.date_range('2000', periods=1), 'datetime64'), (pd.timedelta_range('1', periods=1), 'timedelta64')], ids=lambda x: f'{x}')
def test_pandas_indexing_adapter_non_nanosecond_conversion(index, dtype) -> None:
    data = PandasIndexingAdapter(index.astype(f'{dtype}[s]'))
    with pytest.warns(UserWarning, match='non-nanosecond precision'):
        var = Variable(['time'], data)
    assert var.dtype == np.dtype(f'{dtype}[ns]')