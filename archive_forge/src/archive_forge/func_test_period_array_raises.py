import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import IncompatibleFrequency
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('data, freq, msg', [([pd.Period('2017', 'D'), pd.Period('2017', 'Y')], None, 'Input has different freq'), ([pd.Period('2017', 'D')], 'Y', 'Input has different freq')])
def test_period_array_raises(data, freq, msg):
    with pytest.raises(IncompatibleFrequency, match=msg):
        period_array(data, freq)