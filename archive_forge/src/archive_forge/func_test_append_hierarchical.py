import datetime
from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
def test_append_hierarchical(tmp_path, setup_path, multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data
    df.columns.name = None
    with ensure_clean_store(setup_path) as store:
        store.append('mi', df)
        result = store.select('mi')
        tm.assert_frame_equal(result, df)
        result = store.select('mi', columns=['A', 'B'])
        expected = df.reindex(columns=['A', 'B'])
        tm.assert_frame_equal(result, expected)
    path = tmp_path / 'test.hdf'
    df.to_hdf(path, key='df', format='table')
    result = read_hdf(path, 'df', columns=['A', 'B'])
    expected = df.reindex(columns=['A', 'B'])
    tm.assert_frame_equal(result, expected)