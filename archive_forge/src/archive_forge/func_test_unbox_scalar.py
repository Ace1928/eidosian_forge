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
def test_unbox_scalar(self, arr1d):
    result = arr1d._unbox_scalar(arr1d[0])
    expected = arr1d._ndarray.dtype.type
    assert isinstance(result, expected)
    result = arr1d._unbox_scalar(NaT)
    assert isinstance(result, expected)
    msg = f"'value' should be a {self.scalar_type.__name__}."
    with pytest.raises(ValueError, match=msg):
        arr1d._unbox_scalar('foo')