import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
def test_to_csv_categorical_and_interval(self):
    df = DataFrame({'a': [pd.Interval(Timestamp('2020-01-01'), Timestamp('2020-01-02'), closed='both')]})
    df['a'] = df['a'].astype('category')
    result = df.to_csv()
    expected_rows = [',a', '0,"[2020-01-01 00:00:00, 2020-01-02 00:00:00]"']
    expected = tm.convert_rows_list_to_csv_str(expected_rows)
    assert result == expected