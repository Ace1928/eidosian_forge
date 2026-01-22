import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_survfunc1():
    sr = SurvfuncRight(ti1, st1)
    assert_allclose(sr.surv_prob, surv_prob1, atol=1e-05, rtol=1e-05)
    assert_allclose(sr.surv_prob_se, surv_prob_se1, atol=1e-05, rtol=1e-05)
    assert_allclose(sr.surv_times, times1)
    assert_allclose(sr.n_risk, n_risk1)
    assert_allclose(sr.n_events, n_events1)