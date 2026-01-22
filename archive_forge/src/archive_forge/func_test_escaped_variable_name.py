import numpy as np  # noqa: F401
import pytest
from numpy.testing import assert_equal
from statsmodels.datasets import macrodata
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
def test_escaped_variable_name():
    data = macrodata.load().data
    data.rename(columns={'cpi': 'CPI_'}, inplace=True)
    mod = OLS.from_formula('CPI_ ~ 1 + np.log(realgdp)', data=data)
    res = mod.fit()
    assert 'CPI\\_' in res.summary().as_latex()
    assert 'CPI_' in res.summary().as_text()