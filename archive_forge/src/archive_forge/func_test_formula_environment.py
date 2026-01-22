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
def test_formula_environment():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6]})
    env = patsy.EvalEnvironment([{'z': [3, 6, 9]}])
    model = ols('y ~ x + z', eval_env=env, data=df)
    assert 'z' in model.exog_names
    with pytest.raises(TypeError):
        ols('y ~ x', eval_env='env', data=df)