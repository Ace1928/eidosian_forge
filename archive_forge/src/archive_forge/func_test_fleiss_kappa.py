import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
from statsmodels.stats.inter_rater import (fleiss_kappa, cohens_kappa,
from statsmodels.tools.testing import Holder
def test_fleiss_kappa():
    kappa_wp = 0.21
    assert_almost_equal(fleiss_kappa(table1), kappa_wp, decimal=3)