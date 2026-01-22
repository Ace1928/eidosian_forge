from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_metadata_propagation_indiv_resample(self):
    ts = Series(np.random.default_rng(2).random(1000), index=date_range('20130101', periods=1000, freq='s'), name='foo')
    result = ts.resample('1min').mean()
    tm.assert_metadata_equivalent(ts, result)
    result = ts.resample('1min').min()
    tm.assert_metadata_equivalent(ts, result)
    result = ts.resample('1min').apply(lambda x: x.sum())
    tm.assert_metadata_equivalent(ts, result)