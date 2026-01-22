from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
def test_check_data_stays_on_gpu(toy_weather_data) -> None:
    """Perform some operations and check the data stays on the GPU."""
    freeze = (toy_weather_data['tmin'] <= 0).groupby('time.month').mean('time')
    assert isinstance(freeze.data, cp.ndarray)