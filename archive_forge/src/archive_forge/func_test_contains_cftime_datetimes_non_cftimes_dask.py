from __future__ import annotations
import warnings
from datetime import timedelta
from itertools import product
import numpy as np
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime
from xarray import (
from xarray.coding.times import (
from xarray.coding.variables import SerializationWarning
from xarray.conventions import _update_bounds_attributes, cf_encoder
from xarray.core.common import contains_cftime_datetimes
from xarray.core.utils import is_duck_dask_array
from xarray.testing import assert_equal, assert_identical
from xarray.tests import (
@requires_dask
@pytest.mark.parametrize('non_cftime_data', [DataArray([]), DataArray([1, 2])])
def test_contains_cftime_datetimes_non_cftimes_dask(non_cftime_data) -> None:
    assert not contains_cftime_datetimes(non_cftime_data.variable.chunk())