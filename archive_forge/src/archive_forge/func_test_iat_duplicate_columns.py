import numpy as np
from pandas import (
import pandas._testing as tm
def test_iat_duplicate_columns():
    df = DataFrame([[1, 2]], columns=['x', 'x'])
    assert df.iat[0, 0] == 1