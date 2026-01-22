from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas import Interval
from pandas.core.arrays import IntervalArray
from pandas.tests.extension import base
def test_fillna_non_scalar_raises(data_missing):
    msg = 'can only insert Interval objects and NA into an IntervalArray'
    with pytest.raises(TypeError, match=msg):
        data_missing.fillna([1, 1])