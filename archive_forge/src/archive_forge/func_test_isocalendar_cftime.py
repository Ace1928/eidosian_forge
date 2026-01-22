from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@requires_cftime
def test_isocalendar_cftime(data) -> None:
    with pytest.raises(AttributeError, match="'CFTimeIndex' object has no attribute 'isocalendar'"):
        data.time.dt.isocalendar()