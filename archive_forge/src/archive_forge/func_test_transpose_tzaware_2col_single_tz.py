import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_transpose_tzaware_2col_single_tz(self):
    dti = date_range('2016-04-05 04:30', periods=3, tz='UTC')
    df3 = DataFrame({'A': dti, 'B': dti})
    assert (df3.dtypes == dti.dtype).all()
    res3 = df3.T
    assert (res3.dtypes == dti.dtype).all()