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
def test_tests():
    formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
    dta = load_pandas().data
    results = ols(formula, dta).fit()
    test_formula = '(GNPDEFL = GNP), (UNEMP = 2), (YEAR/1829 = 1)'
    LC = make_hypotheses_matrices(results, test_formula)
    R = LC.coefs
    Q = LC.constants
    npt.assert_almost_equal(R, [[0, 1, -1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1.0 / 1829]], 8)
    npt.assert_array_equal(Q, [[0], [2], [1]])