import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('other', [Timestamp('2016-01-01'), Timestamp('2016-01-01').to_pydatetime(), Timestamp('2016-01-01').to_datetime64(), pd.date_range('2016-01-01', periods=3, freq='h'), pd.date_range('2016-01-01', periods=3, tz='Europe/Brussels'), pd.date_range('2016-01-01', periods=3, freq='s')._data, pd.date_range('2016-01-01', periods=3, tz='Asia/Tokyo')._data, 3.14, np.array([2.0, 3.0, 4.0])])
def test_parr_add_sub_invalid(self, other, box_with_array):
    rng = period_range('1/1/2000', freq='D', periods=3)
    rng = tm.box_expected(rng, box_with_array)
    msg = '|'.join(['(:?cannot add PeriodArray and .*)', '(:?cannot subtract .* from (:?a\\s)?.*)', '(:?unsupported operand type\\(s\\) for \\+: .* and .*)', 'unsupported operand type\\(s\\) for [+-]: .* and .*'])
    assert_invalid_addsub_type(rng, other, msg)
    with pytest.raises(TypeError, match=msg):
        rng + other
    with pytest.raises(TypeError, match=msg):
        other + rng
    with pytest.raises(TypeError, match=msg):
        rng - other
    with pytest.raises(TypeError, match=msg):
        other - rng