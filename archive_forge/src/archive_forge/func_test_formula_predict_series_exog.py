from statsmodels.compat.pandas import assert_series_equal
from io import StringIO
import warnings
import numpy as np
import numpy.testing as npt
import pandas as pd
import patsy
import pytest
from statsmodels.datasets import cpunish
from statsmodels.datasets.longley import load, load_pandas
from statsmodels.formula.api import ols
from statsmodels.formula.formulatools import make_hypotheses_matrices
from statsmodels.tools import add_constant
from statsmodels.tools.testing import assert_equal
def test_formula_predict_series_exog():
    x = np.random.standard_normal((1000, 2))
    data_full = pd.DataFrame(x, columns=['y', 'x'])
    data = data_full.iloc[:500]
    res = ols(formula='y ~ x', data=data).fit()
    oos = data_full.iloc[500:]['x']
    prediction = res.get_prediction(oos)
    assert prediction.predicted_mean.shape[0] == 500