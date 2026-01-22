import numpy as np
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_count_inf_as_na(self):
    ser = Series([pd.Timestamp('1990/1/1')])
    msg = 'use_inf_as_na option is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context('use_inf_as_na', True):
            assert ser.count() == 1