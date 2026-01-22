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
def test_to_csv_dups_cols2(self):
    df = DataFrame(np.ones((5, 3)), index=Index([f'i-{i}' for i in range(5)], name='foo'), columns=Index(['a', 'a', 'b'], dtype=object))
    with tm.ensure_clean() as filename:
        df.to_csv(filename)
        result = read_csv(filename, index_col=0)
        result = result.rename(columns={'a.1': 'a'})
        tm.assert_frame_equal(result, df)