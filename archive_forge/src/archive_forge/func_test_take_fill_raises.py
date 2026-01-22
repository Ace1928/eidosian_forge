from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('fill_value', [2, 2.0, Timestamp(2021, 1, 1, 12).time])
def test_take_fill_raises(self, fill_value, arr1d):
    msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
    with pytest.raises(TypeError, match=msg):
        arr1d.take([0, 1], allow_fill=True, fill_value=fill_value)