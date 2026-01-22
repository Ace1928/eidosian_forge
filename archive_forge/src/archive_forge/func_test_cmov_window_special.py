import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
def test_cmov_window_special(win_types_special, step):
    pytest.importorskip('scipy')
    kwds = {'kaiser': {'beta': 1.0}, 'gaussian': {'std': 1.0}, 'general_gaussian': {'p': 2.0, 'sig': 2.0}, 'exponential': {'tau': 10}}
    vals = np.array([6.95, 15.21, 4.72, 9.12, 13.81, 13.49, 16.68, 9.48, 10.63, 14.48])
    xps = {'gaussian': [np.nan, np.nan, 8.97297, 9.76077, 12.24763, 13.89053, 13.65671, 12.01002, np.nan, np.nan], 'general_gaussian': [np.nan, np.nan, 9.85011, 10.71589, 11.73161, 13.08516, 12.95111, 12.74577, np.nan, np.nan], 'kaiser': [np.nan, np.nan, 9.86851, 11.02969, 11.65161, 12.75129, 12.90702, 12.83757, np.nan, np.nan], 'exponential': [np.nan, np.nan, 9.83364, 11.10472, 11.64551, 12.66138, 12.92379, 12.8377, np.nan, np.nan]}
    xp = Series(xps[win_types_special])[::step]
    rs = Series(vals).rolling(5, win_type=win_types_special, center=True, step=step).mean(**kwds[win_types_special])
    tm.assert_series_equal(xp, rs)