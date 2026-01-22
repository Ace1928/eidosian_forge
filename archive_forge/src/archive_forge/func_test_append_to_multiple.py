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
def test_append_to_multiple(setup_path):
    df1 = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    df2 = df1.copy().rename(columns='{}_2'.format)
    df2['foo'] = 'bar'
    df = concat([df1, df2], axis=1)
    with ensure_clean_store(setup_path) as store:
        msg = 'append_to_multiple requires a selector that is in passed dict'
        with pytest.raises(ValueError, match=msg):
            store.append_to_multiple({'df1': ['A', 'B'], 'df2': None}, df, selector='df3')
        with pytest.raises(ValueError, match=msg):
            store.append_to_multiple({'df1': None, 'df2': None}, df, selector='df3')
        msg = 'append_to_multiple must have a dictionary specified as the way to split the value'
        with pytest.raises(ValueError, match=msg):
            store.append_to_multiple('df1', df, 'df1')
        store.append_to_multiple({'df1': ['A', 'B'], 'df2': None}, df, selector='df1')
        result = store.select_as_multiple(['df1', 'df2'], where=['A>0', 'B>0'], selector='df1')
        expected = df[(df.A > 0) & (df.B > 0)]
        tm.assert_frame_equal(result, expected)