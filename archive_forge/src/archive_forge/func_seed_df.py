from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def seed_df(seed_nans, n, m):
    np.random.seed(1234)
    days = date_range('2015-08-24', periods=10)
    frame = DataFrame({'1st': np.random.choice(list('abcd'), n), '2nd': np.random.choice(days, n), '3rd': np.random.randint(1, m + 1, n)})
    if seed_nans:
        frame['3rd'] = frame['3rd'].astype('float')
        frame.loc[1::11, '1st'] = np.nan
        frame.loc[3::17, '2nd'] = np.nan
        frame.loc[7::19, '3rd'] = np.nan
        frame.loc[8::19, '3rd'] = np.nan
        frame.loc[9::19, '3rd'] = np.nan
    return frame