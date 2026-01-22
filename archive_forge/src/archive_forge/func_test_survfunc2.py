import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_survfunc2():
    sr = SurvfuncRight(ti2, st2)
    assert_allclose(sr.surv_prob, surv_prob2, atol=1e-05, rtol=1e-05)
    assert_allclose(sr.surv_prob_se, surv_prob_se2, atol=1e-05, rtol=1e-05)
    assert_allclose(sr.surv_times, times2)
    assert_allclose(sr.n_risk, n_risk2)
    assert_allclose(sr.n_events, n_events2)