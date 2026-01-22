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
def test_formula_predict():
    from numpy import log
    formula = 'TOTEMP ~ log(GNPDEFL) + log(GNP) + UNEMP + ARMED +\n                    POP + YEAR'
    data = load_pandas()
    dta = load_pandas().data
    results = ols(formula, dta).fit()
    npt.assert_almost_equal(results.fittedvalues.values, results.predict(data.exog), 8)