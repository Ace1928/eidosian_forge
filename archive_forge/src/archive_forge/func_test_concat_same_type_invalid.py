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
def test_concat_same_type_invalid(self, arr1d):
    arr = arr1d
    if arr.tz is None:
        other = arr.tz_localize('UTC')
    else:
        other = arr.tz_localize(None)
    with pytest.raises(ValueError, match='to_concat must have the same'):
        arr._concat_same_type([arr, other])