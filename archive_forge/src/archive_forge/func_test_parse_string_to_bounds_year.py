from __future__ import annotations
import pickle
from datetime import timedelta
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import (
from xarray.tests import (
from xarray.tests.test_coding_times import (
@requires_cftime
def test_parse_string_to_bounds_year(date_type, dec_days):
    parsed = date_type(2, 2, 10, 6, 2, 8, 1)
    expected_start = date_type(2, 1, 1)
    expected_end = date_type(2, 12, dec_days, 23, 59, 59, 999999)
    result_start, result_end = _parsed_string_to_bounds(date_type, 'year', parsed)
    assert result_start == expected_start
    assert result_end == expected_end