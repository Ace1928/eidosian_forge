import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_subclass_stack(self):
    df = tm.SubclassedDataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['a', 'b', 'c'], columns=['X', 'Y', 'Z'])
    res = df.stack(future_stack=True)
    exp = tm.SubclassedSeries([1, 2, 3, 4, 5, 6, 7, 8, 9], index=[list('aaabbbccc'), list('XYZXYZXYZ')])
    tm.assert_series_equal(res, exp)