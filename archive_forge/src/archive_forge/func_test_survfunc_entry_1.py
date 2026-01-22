import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_survfunc_entry_1():
    times = np.r_[1, 3, 3, 5, 5, 7, 7, 8, 8, 9, 10, 10]
    status = np.r_[1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]
    entry = np.r_[0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 0]
    sf = SurvfuncRight(times, status, entry=entry)
    assert_allclose(sf.n_risk, np.r_[2, 6, 9, 7, 5, 3, 2])
    assert_allclose(sf.surv_times, np.r_[1, 3, 5, 7, 8, 9, 10])
    assert_allclose(sf.surv_prob, np.r_[0.5, 0.4167, 0.3241, 0.2778, 0.2222, 0.1481, 0.0741], atol=0.0001)
    assert_allclose(sf.surv_prob_se, np.r_[0.3536, 0.3043, 0.2436, 0.2132, 0.1776, 0.133, 0.0846], atol=0.0001)