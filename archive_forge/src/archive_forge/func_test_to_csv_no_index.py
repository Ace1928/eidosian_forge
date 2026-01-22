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
def test_to_csv_no_index(self):
    with tm.ensure_clean('__tmp_to_csv_no_index__') as path:
        df = DataFrame({'c1': [1, 2, 3], 'c2': [4, 5, 6]})
        df.to_csv(path, index=False)
        result = read_csv(path)
        tm.assert_frame_equal(df, result)
        df['c3'] = Series([7, 8, 9], dtype='int64')
        df.to_csv(path, index=False)
        result = read_csv(path)
        tm.assert_frame_equal(df, result)