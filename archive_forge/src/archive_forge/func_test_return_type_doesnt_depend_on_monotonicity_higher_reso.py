from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_return_type_doesnt_depend_on_monotonicity_higher_reso(self):
    dti = date_range(start='2015-5-13 23:59:00', freq='min', periods=3)
    ser = Series(range(3), index=dti)
    ser2 = Series(range(3), index=[dti[1], dti[0], dti[2]])
    key = '2015-5-14 00:00:00'
    result = ser.loc[key]
    assert result == 1
    result = ser.iloc[::-1].loc[key]
    assert result == 1
    result2 = ser2.loc[key]
    assert result2 == 0