from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
def test_not_datetime_type(self) -> None:
    nontime_data = self.data.copy()
    int_data = np.arange(len(self.data.time)).astype('int8')
    nontime_data = nontime_data.assign_coords(time=int_data)
    with pytest.raises(AttributeError, match='dt'):
        nontime_data.time.dt