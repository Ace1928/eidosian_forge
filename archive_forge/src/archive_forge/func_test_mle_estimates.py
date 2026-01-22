import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
def test_mle_estimates(self):
    start_params = [0.5, 0.4, 4, 32, 2.3, -2, -9]
    mle_res = self.res.model.fit(start_params, disp=0, maxiter=100)
    assert_(self.res.llf <= mle_res.llf)