import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
def test_weightstats_2d_w1():
    x1 = [[1], [2]]
    w1 = [[1], [2]]
    d1 = DescrStatsW(x1, w1)
    print(len(np.array(w1).shape))
    assert (d1.quantile([0.5, 1.0]) == 2).all().all()