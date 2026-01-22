from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@pytest.fixture()
def cftime_date_type(calendar):
    from xarray.tests.test_coding_times import _all_cftime_date_types
    return _all_cftime_date_types()[calendar]