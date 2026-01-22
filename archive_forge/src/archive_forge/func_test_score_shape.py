import numpy as np
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
from numpy.testing import assert_, assert_raises, assert_equal, assert_allclose
def test_score_shape():
    endog = macrodata['infl']
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))
    with mod.fix_params({'ar.L1': 0.5}):
        score = mod.score([1.0])
    assert_equal(score.shape, (1,))