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
def test_to_csv_cols_reordering(self):
    chunksize = 5
    N = int(chunksize * 2.5)
    df = DataFrame(np.ones((N, 3)), index=Index([f'i-{i}' for i in range(N)], name='a'), columns=Index([f'i-{i}' for i in range(3)], name='a'))
    cs = df.columns
    cols = [cs[2], cs[0]]
    with tm.ensure_clean() as path:
        df.to_csv(path, columns=cols, chunksize=chunksize)
        rs_c = read_csv(path, index_col=0)
    tm.assert_frame_equal(df[cols], rs_c, check_names=False)