import itertools
import os
import numpy as np
from statsmodels.duration.hazard_regression import PHReg
from numpy.testing import (assert_allclose,
import pandas as pd
import pytest
from .results import survival_r_results
from .results import survival_enet_r_results
def test_predict_formula(self):
    n = 100
    np.random.seed(34234)
    time = 50 * np.random.uniform(size=n)
    status = np.random.randint(0, 2, n).astype(np.float64)
    exog = np.random.uniform(1, 2, size=(n, 2))
    df = pd.DataFrame({'time': time, 'status': status, 'exog1': exog[:, 0], 'exog2': exog[:, 1]})
    fml = 'time ~ exog1 + np.log(exog2) + exog1*exog2'
    model1 = PHReg.from_formula(fml, df, status=status)
    result1 = model1.fit()
    from patsy import dmatrix
    dfp = dmatrix(model1.data.design_info, df)
    pr1 = result1.predict()
    pr2 = result1.predict(exog=df)
    pr3 = model1.predict(result1.params, exog=dfp)
    pr4 = model1.predict(result1.params, cov_params=result1.cov_params(), exog=dfp)
    prl = (pr1, pr2, pr3, pr4)
    for i in range(4):
        for j in range(i):
            assert_allclose(prl[i].predicted_values, prl[j].predicted_values)
    prl = (pr1, pr2, pr4)
    for i in range(3):
        for j in range(i):
            assert_allclose(prl[i].standard_errors, prl[j].standard_errors)