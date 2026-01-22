from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
import pandas._testing as tm
def test_invert_empty_not_input(self):
    df = pd.DataFrame()
    result = ~df
    tm.assert_frame_equal(df, result)
    assert df is not result