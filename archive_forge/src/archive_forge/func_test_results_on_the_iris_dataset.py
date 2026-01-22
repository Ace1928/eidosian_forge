import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import get_rdataset
from statsmodels.datasets.tests.test_utils import IGNORED_EXCEPTIONS
import statsmodels.stats.dist_dependence_measures as ddm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def test_results_on_the_iris_dataset(self):
    """
        R code example from the `energy` package documentation for
        `energy::distance_covariance.test`:

        > x <- iris[1:50, 1:4]
        > y <- iris[51:100, 1:4]
        > set.seed(1)
        > dcov.test(x, y, R=200)

            dCov independence test (permutation test)

        data:  index 1, replicates 200
        nV^2 = 0.5254, p-value = 0.9552
        sample estimates:
             dCov
        0.1025087
        """
    try:
        iris = get_rdataset('iris').data.values[:, :4]
    except IGNORED_EXCEPTIONS:
        pytest.skip('Failed with HTTPError or URLError, these are random')
    x = np.asarray(iris[:50], dtype=float)
    y = np.asarray(iris[50:100], dtype=float)
    stats = ddm.distance_statistics(x, y)
    assert_almost_equal(stats.test_statistic, 0.5254, 4)
    assert_almost_equal(stats.distance_correlation, 0.3060479, 4)
    assert_almost_equal(stats.distance_covariance, 0.1025087, 4)
    assert_almost_equal(stats.dvar_x, 0.2712927, 4)
    assert_almost_equal(stats.dvar_y, 0.4135274, 4)
    assert_almost_equal(stats.S, 0.667456, 4)
    test_statistic, _, method = ddm.distance_covariance_test(x, y, B=199)
    assert_almost_equal(test_statistic, 0.5254, 4)
    assert method == 'emp'