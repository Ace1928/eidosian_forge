import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_melt_mixed_int_str_value_vars(self):
    df = DataFrame({0: ['foo'], 'a': ['bar']})
    result = melt(df, value_vars=[0, 'a'])
    expected = DataFrame({'variable': [0, 'a'], 'value': ['foo', 'bar']})
    tm.assert_frame_equal(result, expected)