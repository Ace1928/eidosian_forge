import itertools
import os
import numpy as np
from statsmodels.duration.hazard_regression import PHReg
from numpy.testing import (assert_allclose,
import pandas as pd
import pytest
from .results import survival_r_results
from .results import survival_enet_r_results
def test_formula_args(self):
    np.random.seed(34234)
    n = 200
    time = 50 * np.random.uniform(size=n)
    status = np.random.randint(0, 2, size=n).astype(np.float64)
    exog = np.random.normal(size=(200, 2))
    offset = np.random.uniform(size=n)
    entry = np.random.uniform(0, 1, size=n) * time
    df = pd.DataFrame({'time': time, 'status': status, 'x1': exog[:, 0], 'x2': exog[:, 1], 'offset': offset, 'entry': entry})
    model1 = PHReg.from_formula('time ~ x1 + x2', status='status', offset='offset', entry='entry', data=df)
    result1 = model1.fit()
    model2 = PHReg.from_formula('time ~ x1 + x2', status=df.status, offset=df.offset, entry=df.entry, data=df)
    result2 = model2.fit()
    assert_allclose(result1.params, result2.params)
    assert_allclose(result1.bse, result2.bse)