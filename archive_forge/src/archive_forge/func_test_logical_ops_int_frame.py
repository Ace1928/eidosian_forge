import operator
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_logical_ops_int_frame(self):
    df1a_int = DataFrame(1, index=[1], columns=['A'])
    df1a_bool = DataFrame(True, index=[1], columns=['A'])
    result = df1a_int | df1a_bool
    tm.assert_frame_equal(result, df1a_bool)
    res_ser = df1a_int['A'] | df1a_bool['A']
    tm.assert_series_equal(res_ser, df1a_bool['A'])