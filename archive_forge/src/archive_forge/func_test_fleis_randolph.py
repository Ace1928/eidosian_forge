import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
from statsmodels.stats.inter_rater import (fleiss_kappa, cohens_kappa,
from statsmodels.tools.testing import Holder
def test_fleis_randolph():
    table = [[7, 0], [7, 0]]
    assert_equal(fleiss_kappa(table, method='unif'), 1)
    table = [[6.99, 0.01], [6.99, 0.01]]
    assert_allclose(fleiss_kappa(table), -0.166667, atol=6e-06)
    assert_allclose(fleiss_kappa(table, method='unif'), 0.993343, atol=6e-06)
    table = [[7, 1], [3, 5]]
    assert_allclose(fleiss_kappa(table, method='fleiss'), 0.161905, atol=6e-06)
    assert_allclose(fleiss_kappa(table, method='randolph'), 0.214286, atol=6e-06)
    table = [[7, 0], [0, 7]]
    assert_allclose(fleiss_kappa(table), 1)
    assert_allclose(fleiss_kappa(table, method='uniform'), 1)
    table = [[6, 1, 0], [0, 7, 0]]
    assert_allclose(fleiss_kappa(table), 0.708333, atol=6e-06)
    assert_allclose(fleiss_kappa(table, method='rand'), 0.785714, atol=6e-06)