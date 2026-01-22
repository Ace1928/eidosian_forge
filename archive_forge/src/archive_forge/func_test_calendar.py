from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
def test_calendar(self) -> None:
    cal = self.data.time.dt.calendar
    assert cal == 'proleptic_gregorian'