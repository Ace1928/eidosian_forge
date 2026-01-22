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
def test_to_csv_mixed(self):

    def create_cols(name):
        return [f'{name}{i:03d}' for i in range(5)]
    df_float = DataFrame(np.random.default_rng(2).standard_normal((100, 5)), dtype='float64', columns=create_cols('float'))
    df_int = DataFrame(np.random.default_rng(2).standard_normal((100, 5)).astype('int64'), dtype='int64', columns=create_cols('int'))
    df_bool = DataFrame(True, index=df_float.index, columns=create_cols('bool'))
    df_object = DataFrame('foo', index=df_float.index, columns=create_cols('object'))
    df_dt = DataFrame(Timestamp('20010101').as_unit('ns'), index=df_float.index, columns=create_cols('date'))
    df_float.iloc[30:50, 1:3] = np.nan
    df_dt.iloc[30:50, 1:3] = np.nan
    df = pd.concat([df_float, df_int, df_bool, df_object, df_dt], axis=1)
    dtypes = {}
    for n, dtype in [('float', np.float64), ('int', np.int64), ('bool', np.bool_), ('object', object)]:
        for c in create_cols(n):
            dtypes[c] = dtype
    with tm.ensure_clean() as filename:
        df.to_csv(filename)
        rs = read_csv(filename, index_col=0, dtype=dtypes, parse_dates=create_cols('date'))
        tm.assert_frame_equal(rs, df)