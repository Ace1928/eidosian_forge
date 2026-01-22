import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
@pytest.mark.parametrize('f,xp', [('mean', [[np.nan, np.nan], [np.nan, np.nan], [9.252, 9.392], [8.644, 9.906], [8.87, 10.208], [6.81, 8.588], [7.792, 8.644], [9.05, 7.824], [np.nan, np.nan], [np.nan, np.nan]]), ('std', [[np.nan, np.nan], [np.nan, np.nan], [3.789706, 4.068313], [3.429232, 3.237411], [3.589269, 3.22081], [3.405195, 2.380655], [3.281839, 2.369869], [3.676846, 1.801799], [np.nan, np.nan], [np.nan, np.nan]]), ('var', [[np.nan, np.nan], [np.nan, np.nan], [14.36187, 16.55117], [11.75963, 10.48083], [12.88285, 10.37362], [11.59535, 5.66752], [10.77047, 5.61628], [13.5192, 3.24648], [np.nan, np.nan], [np.nan, np.nan]]), ('sum', [[np.nan, np.nan], [np.nan, np.nan], [46.26, 46.96], [43.22, 49.53], [44.35, 51.04], [34.05, 42.94], [38.96, 43.22], [45.25, 39.12], [np.nan, np.nan], [np.nan, np.nan]])])
def test_cmov_window_frame(f, xp, step):
    pytest.importorskip('scipy')
    df = DataFrame(np.array([[12.18, 3.64], [10.18, 9.16], [13.24, 14.61], [4.51, 8.11], [6.15, 11.44], [9.14, 6.21], [11.31, 10.67], [2.94, 6.51], [9.42, 8.39], [12.44, 7.34]]))
    xp = DataFrame(np.array(xp))[::step]
    roll = df.rolling(5, win_type='boxcar', center=True, step=step)
    rs = getattr(roll, f)()
    tm.assert_frame_equal(xp, rs)