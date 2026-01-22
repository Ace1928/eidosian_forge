import itertools
import os
import numpy as np
from statsmodels.duration.hazard_regression import PHReg
from numpy.testing import (assert_allclose,
import pandas as pd
import pytest
from .results import survival_r_results
from .results import survival_enet_r_results
def test_formula_cat_interactions(self):
    time = np.r_[1, 2, 3, 4, 5, 6, 7, 8, 9]
    status = np.r_[1, 1, 0, 0, 1, 0, 1, 1, 1]
    x1 = np.r_[1, 1, 1, 2, 2, 2, 3, 3, 3]
    x2 = np.r_[1, 2, 3, 1, 2, 3, 1, 2, 3]
    df = pd.DataFrame({'time': time, 'status': status, 'x1': x1, 'x2': x2})
    model1 = PHReg.from_formula('time ~ C(x1) + C(x2) + C(x1)*C(x2)', status='status', data=df)
    assert_equal(model1.exog.shape, [9, 8])