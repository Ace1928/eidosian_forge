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
def test_append_to_multiple_dropna_false(setup_path):
    df1 = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    df2 = df1.copy().rename(columns='{}_2'.format)
    df1.iloc[1, df1.columns.get_indexer(['A', 'B'])] = np.nan
    df = concat([df1, df2], axis=1)
    with ensure_clean_store(setup_path) as store, pd.option_context('io.hdf.dropna_table', True):
        store.append_to_multiple({'df1a': ['A', 'B'], 'df2a': None}, df, selector='df1a', dropna=False)
        msg = 'all tables must have exactly the same nrows!'
        with pytest.raises(ValueError, match=msg):
            store.select_as_multiple(['df1a', 'df2a'])
        assert not store.select('df1a').index.equals(store.select('df2a').index)