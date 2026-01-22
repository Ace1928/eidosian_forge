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
def test_binop_maybe_preserve_name(self, datetime_series):
    result = datetime_series * datetime_series
    assert result.name == datetime_series.name
    result = datetime_series.mul(datetime_series)
    assert result.name == datetime_series.name
    result = datetime_series * datetime_series[:-2]
    assert result.name == datetime_series.name
    cp = datetime_series.copy()
    cp.name = 'something else'
    result = datetime_series + cp
    assert result.name is None
    result = datetime_series.add(cp)
    assert result.name is None
    ops = ['add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow']
    ops = ops + ['r' + op for op in ops]
    for op in ops:
        ser = datetime_series.copy()
        result = getattr(ser, op)(ser)
        assert result.name == datetime_series.name
        cp = datetime_series.copy()
        cp.name = 'changed'
        result = getattr(ser, op)(cp)
        assert result.name is None