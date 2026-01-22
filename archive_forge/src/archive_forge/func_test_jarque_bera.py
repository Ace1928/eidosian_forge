import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_jarque_bera():
    st_pv_R = np.array([1.966267722686169, 0.3741367669648314])
    jb = jarque_bera(x)[:2]
    assert_almost_equal(jb, st_pv_R, 14)
    st_pv_R = np.array([78.329987305556, 0.0])
    jb = jarque_bera(x ** 2)[:2]
    assert_almost_equal(jb, st_pv_R, 13)
    st_pv_R = np.array([5.713575079670667, 0.0574530296971343])
    jb = jarque_bera(np.log(x ** 2))[:2]
    assert_almost_equal(jb, st_pv_R, 14)
    st_pv_R = np.array([2.648931574849576, 0.2659449923067881])
    jb = jarque_bera(np.exp(-x ** 2))[:2]
    assert_almost_equal(jb, st_pv_R, 14)