import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tsa.regime_switching import markov_switching
def check_transtion_2(params):
    assert_equal(params['regime_transition'], np.s_[0:2])
    assert_equal(params[0, 'regime_transition'], [0])
    assert_equal(params[1, 'regime_transition'], [1])
    assert_equal(params['regime_transition', 0], [0])
    assert_equal(params['regime_transition', 1], [1])