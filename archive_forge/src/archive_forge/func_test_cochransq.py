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
def test_cochransq():
    x = np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1]])
    res_qstat = 2.8
    res_pvalue = 0.246597
    res = cochrans_q(x)
    assert_almost_equal([res.statistic, res.pvalue], [res_qstat, res_pvalue])
    a, b = x[:, :2].T
    res = cochrans_q(x[:, :2])
    with pytest.warns(FutureWarning):
        assert_almost_equal(sbmcnemar(a, b, exact=False, correction=False), [res.statistic, res.pvalue])