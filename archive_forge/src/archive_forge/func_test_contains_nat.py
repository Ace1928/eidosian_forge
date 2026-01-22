from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_contains_nat(self):
    idx = period_range('2007-01', freq='M', periods=10)
    assert NaT not in idx
    assert None not in idx
    assert float('nan') not in idx
    assert np.nan not in idx
    idx = PeriodIndex(['2011-01', 'NaT', '2011-02'], freq='M')
    assert NaT in idx
    assert None in idx
    assert float('nan') in idx
    assert np.nan in idx