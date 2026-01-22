from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
@pytest.mark.parametrize('use_flox', [True, False])
def test_min_count_error(use_flox: bool) -> None:
    if use_flox and (not has_flox):
        pytest.skip()
    da = DataArray(data=np.array([np.nan, 1, 1, np.nan, 1, 1]), dims='x', coords={'labels': ('x', np.array([1, 2, 3, 1, 2, 3]))})
    with xr.set_options(use_flox=use_flox):
        with pytest.raises(TypeError):
            da.groupby('labels').mean(min_count=1)