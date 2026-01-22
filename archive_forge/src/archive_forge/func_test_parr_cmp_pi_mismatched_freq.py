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
@pytest.mark.parametrize('freq', ['M', '2M', '3M'])
def test_parr_cmp_pi_mismatched_freq(self, freq, box_with_array):
    base = PeriodIndex(['2011-01', '2011-02', '2011-03', '2011-04'], freq=freq)
    base = tm.box_expected(base, box_with_array)
    msg = f'Invalid comparison between dtype=period\\[{freq}\\] and Period'
    with pytest.raises(TypeError, match=msg):
        base <= Period('2011', freq='Y')
    with pytest.raises(TypeError, match=msg):
        Period('2011', freq='Y') >= base
    idx = PeriodIndex(['2011', '2012', '2013', '2014'], freq='Y')
    rev_msg = 'Invalid comparison between dtype=period\\[Y-DEC\\] and PeriodArray'
    idx_msg = rev_msg if box_with_array in [tm.to_array, pd.array] else msg
    with pytest.raises(TypeError, match=idx_msg):
        base <= idx
    msg = f'Invalid comparison between dtype=period\\[{freq}\\] and Period'
    with pytest.raises(TypeError, match=msg):
        base <= Period('2011', freq='4M')
    with pytest.raises(TypeError, match=msg):
        Period('2011', freq='4M') >= base
    idx = PeriodIndex(['2011', '2012', '2013', '2014'], freq='4M')
    rev_msg = 'Invalid comparison between dtype=period\\[4M\\] and PeriodArray'
    idx_msg = rev_msg if box_with_array in [tm.to_array, pd.array] else msg
    with pytest.raises(TypeError, match=idx_msg):
        base <= idx