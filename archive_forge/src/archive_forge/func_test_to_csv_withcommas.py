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
def test_to_csv_withcommas(self):
    df = DataFrame({'A': [1, 2, 3], 'B': ['5,6', '7,8', '9,0']})
    with tm.ensure_clean('__tmp_to_csv_withcommas__.csv') as path:
        df.to_csv(path)
        df2 = self.read_csv(path)
        tm.assert_frame_equal(df2, df)