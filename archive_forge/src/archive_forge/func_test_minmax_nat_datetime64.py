from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('op', ['min', 'max'])
def test_minmax_nat_datetime64(self, op):
    obj = DatetimeIndex([])
    assert isna(getattr(obj, op)())
    obj = DatetimeIndex([NaT])
    assert isna(getattr(obj, op)())
    obj = DatetimeIndex([NaT, NaT, NaT])
    assert isna(getattr(obj, op)())