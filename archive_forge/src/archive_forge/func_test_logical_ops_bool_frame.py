import operator
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_logical_ops_bool_frame(self):
    df1a_bool = DataFrame(True, index=[1], columns=['A'])
    result = df1a_bool & df1a_bool
    tm.assert_frame_equal(result, df1a_bool)
    result = df1a_bool | df1a_bool
    tm.assert_frame_equal(result, df1a_bool)