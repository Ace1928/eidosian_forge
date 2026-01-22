from datetime import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.core.arrays import (
@pytest.mark.parametrize('other', [Timedelta(0), Timedelta(0).to_pytimedelta(), pytest.param(Timedelta(0).to_timedelta64(), marks=pytest.mark.xfail(not np_version_gte1p24p3, reason="td64 doesn't return NotImplemented, see numpy#17017")), Timestamp(0), Timestamp(0).to_pydatetime(), pytest.param(Timestamp(0).to_datetime64(), marks=pytest.mark.xfail(not np_version_gte1p24p3, reason="dt64 doesn't return NotImplemented, see numpy#17017")), Timestamp(0).tz_localize('UTC'), NaT])
def test_nat_comparisons(compare_operators_no_eq_ne, other):
    opname = compare_operators_no_eq_ne
    assert getattr(NaT, opname)(other) is False
    op = getattr(operator, opname.strip('_'))
    assert op(NaT, other) is False
    assert op(other, NaT) is False