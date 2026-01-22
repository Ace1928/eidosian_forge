import os
import numpy as np
from numpy.testing import (
import pytest
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
def test_exog_predict(self):
    rfile = os.path.join(rpath, 'test_lowess_simple.csv')
    test_data = np.genfromtxt(open(rfile, 'rb'), delimiter=',', names=True)
    y, x = (test_data['y'], test_data['x'])
    target = lowess(y, x, is_sorted=True)
    perm_idx = np.arange(len(x) // 2)
    np.random.shuffle(perm_idx)
    actual_lowess = lowess(y, x, xvals=x[perm_idx], missing='none')
    assert_almost_equal(target[perm_idx, 1], actual_lowess, decimal=13)
    target_it0 = lowess(y, x, return_sorted=False, it=0)
    actual_lowess2 = lowess(y, x, xvals=x[perm_idx], it=0)
    assert_almost_equal(target_it0[perm_idx], actual_lowess2, decimal=13)
    with pytest.raises(ValueError):
        lowess(y, x, xvals=np.array([np.nan, 5, 3]), missing='raise')
    actual_lowess3 = lowess(y, x, xvals=x, is_sorted=True)
    assert_equal(actual_lowess3, target[:, 1])
    y[[5, 6]] = np.nan
    x[3] = np.nan
    target = lowess(y, x, is_sorted=True)
    perm_idx = np.arange(target.shape[0])
    actual_lowess1 = lowess(y, x, xvals=target[perm_idx, 0])
    assert_almost_equal(target[perm_idx, 1], actual_lowess1, decimal=13)
    actual_lowess2 = lowess(y, x, xvals=x, missing='drop')
    all_finite = np.isfinite(x) & np.isfinite(y)
    assert_equal(actual_lowess2[all_finite], target[:, 1])
    with pytest.raises(ValueError):
        lowess(y, x, xvals=np.array([[5], [10]]))