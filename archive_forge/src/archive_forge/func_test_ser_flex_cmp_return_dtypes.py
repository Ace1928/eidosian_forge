from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
@pytest.mark.parametrize('opname', ['eq', 'ne', 'gt', 'lt', 'ge', 'le'])
def test_ser_flex_cmp_return_dtypes(self, opname):
    ser = Series([1, 3, 2], index=range(3))
    const = 2
    result = getattr(ser, opname)(const).dtypes
    expected = np.dtype('bool')
    assert result == expected