from itertools import product
import numpy as np
import pytest
from pandas._libs import lib
import pandas as pd
import pandas._testing as tm
def test_convert_byte_string_dtype(self):
    byte_str = b'binary-string'
    df = pd.DataFrame(data={'A': byte_str}, index=[0])
    result = df.convert_dtypes()
    expected = df
    tm.assert_frame_equal(result, expected)