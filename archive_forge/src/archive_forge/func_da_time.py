from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@pytest.fixture
def da_time():
    return xr.DataArray([np.nan, 1, 2, np.nan, np.nan, 5, np.nan, np.nan, np.nan, np.nan, 10], dims=['t'])