from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
def test_unstack_nan_index5(self):
    df = DataFrame({'1st': [1, 2, 1, 2, 1, 2], '2nd': date_range('2014-02-01', periods=6, freq='D'), 'jim': 100 + np.arange(6), 'joe': (np.random.default_rng(2).standard_normal(6) * 10).round(2)})
    df['3rd'] = df['2nd'] - pd.Timestamp('2014-02-02')
    df.loc[1, '2nd'] = df.loc[3, '2nd'] = np.nan
    df.loc[1, '3rd'] = df.loc[4, '3rd'] = np.nan
    left = df.set_index(['1st', '2nd', '3rd']).unstack(['2nd', '3rd'])
    assert left.notna().values.sum() == 2 * len(df)
    for col in ['jim', 'joe']:
        for _, r in df.iterrows():
            key = (r['1st'], (col, r['2nd'], r['3rd']))
            assert r[col] == left.loc[key]