from statsmodels.compat.pandas import (
from datetime import datetime
import numpy as np
from numpy import array, column_stack
from numpy.testing import (
from pandas import DataFrame, concat, date_range
from statsmodels.datasets import macrodata
from statsmodels.tsa.filters._utils import pandas_wrapper
from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.cf_filter import cffilter
from statsmodels.tsa.filters.filtertools import (
from statsmodels.tsa.filters.hp_filter import hpfilter
def test_bking_pandas():
    dta = macrodata.load_pandas().data
    index = date_range(start='1959-01-01', end='2009-10-01', freq=QUARTER_END)
    dta.index = index
    filtered = bkfilter(dta['infl'])
    nd_filtered = bkfilter(dta['infl'].values)
    assert_equal(filtered.values, nd_filtered)
    assert_equal(filtered.index[0], datetime(1962, 3, 31))
    assert_equal(filtered.index[-1], datetime(2006, 9, 30))
    assert_equal(filtered.name, 'infl_cycle')
    filtered = bkfilter(dta[['infl', 'unemp']])
    nd_filtered = bkfilter(dta[['infl', 'unemp']].values)
    assert_equal(filtered.values, nd_filtered)
    assert_equal(filtered.index[0], datetime(1962, 3, 31))
    assert_equal(filtered.index[-1], datetime(2006, 9, 30))
    assert_equal(filtered.columns.values, ['infl_cycle', 'unemp_cycle'])