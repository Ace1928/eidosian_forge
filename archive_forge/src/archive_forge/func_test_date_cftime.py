from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@requires_cftime
def test_date_cftime(data) -> None:
    with pytest.raises(AttributeError, match="'CFTimeIndex' object has no attribute `date`. Consider using the floor method instead, for instance: `.time.dt.floor\\('D'\\)`."):
        data.time.dt.date()