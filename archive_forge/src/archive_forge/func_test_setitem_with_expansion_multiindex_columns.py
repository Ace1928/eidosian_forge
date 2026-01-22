import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_with_expansion_multiindex_columns(self, multiindex_year_month_day_dataframe_random_data):
    ymd = multiindex_year_month_day_dataframe_random_data
    df = ymd[:5].T
    df[2000, 1, 10] = df[2000, 1, 7]
    assert isinstance(df.columns, MultiIndex)
    assert (df[2000, 1, 10] == df[2000, 1, 7]).all()