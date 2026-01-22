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
def test_get_loc(date_type, index):
    result = index.get_loc('0001')
    assert result == slice(0, 2)
    result = index.get_loc(date_type(1, 2, 1))
    assert result == 1
    result = index.get_loc('0001-02-01')
    assert result == slice(1, 2)
    with pytest.raises(KeyError, match='1234'):
        index.get_loc('1234')