from statsmodels.compat.python import lzip
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
from scipy import stats
import pytest
from statsmodels.stats.contingency_tables import (
from statsmodels.sandbox.stats.runs import (Runs,
from statsmodels.sandbox.stats.runs import mcnemar as sbmcnemar
from statsmodels.stats.nonparametric import (
from statsmodels.tools.testing import Holder
def test_cochransq3():
    dt = [('A', 'S1'), ('B', 'S1'), ('C', 'S1'), ('count', int)]
    dta = np.array([('F', 'F', 'F', 6), ('U', 'F', 'F', 2), ('F', 'F', 'U', 16), ('U', 'F', 'U', 4), ('F', 'U', 'F', 2), ('U', 'U', 'F', 6), ('F', 'U', 'U', 4), ('U', 'U', 'U', 6)], dt)
    cases = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]])
    count = np.array([6, 2, 16, 4, 2, 6, 4, 6])
    data = np.repeat(cases, count, 0)
    res = cochrans_q(data)
    assert_allclose([res.statistic, res.pvalue], [8.4706, 0.0145], atol=5e-05)