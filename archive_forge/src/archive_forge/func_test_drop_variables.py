from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_drop_variables(self) -> None:
    data = create_test_data()
    assert_identical(data, data.drop_vars([]))
    expected = Dataset({k: data[k] for k in data.variables if k != 'time'})
    actual = data.drop_vars('time')
    assert_identical(expected, actual)
    actual = data.drop_vars(['time'])
    assert_identical(expected, actual)
    with pytest.raises(ValueError, match=re.escape("These variables cannot be found in this dataset: ['not_found_here']")):
        data.drop_vars('not_found_here')
    actual = data.drop_vars('not_found_here', errors='ignore')
    assert_identical(data, actual)
    actual = data.drop_vars(['not_found_here'], errors='ignore')
    assert_identical(data, actual)
    actual = data.drop_vars(['time', 'not_found_here'], errors='ignore')
    assert_identical(expected, actual)
    with pytest.warns(DeprecationWarning):
        actual = data.drop('not_found_here', errors='ignore')
    assert_identical(data, actual)
    with pytest.warns(DeprecationWarning):
        actual = data.drop(['not_found_here'], errors='ignore')
    assert_identical(data, actual)
    with pytest.warns(DeprecationWarning):
        actual = data.drop(['time', 'not_found_here'], errors='ignore')
    assert_identical(expected, actual)
    with pytest.warns(DeprecationWarning):
        actual = data.drop({'time', 'not_found_here'}, errors='ignore')
    assert_identical(expected, actual)