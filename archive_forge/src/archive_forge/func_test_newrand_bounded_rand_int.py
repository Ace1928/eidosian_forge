import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm._bounds import l1_min_c
from sklearn.svm._newrand import bounded_rand_int_wrap, set_seed_wrap
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('range_, n_pts', [(_MAX_UNSIGNED_INT, 10000), (100, 25)])
def test_newrand_bounded_rand_int(range_, n_pts):
    """Test that `bounded_rand_int` follows a uniform distribution"""
    set_seed_wrap(42)
    n_iter = 100
    ks_pvals = []
    uniform_dist = stats.uniform(loc=0, scale=range_)
    for _ in range(n_iter):
        sample = [bounded_rand_int_wrap(range_) for _ in range(n_pts)]
        res = stats.kstest(sample, uniform_dist.cdf)
        ks_pvals.append(res.pvalue)
    uniform_p_vals_dist = stats.uniform(loc=0, scale=1)
    res_pvals = stats.kstest(ks_pvals, uniform_p_vals_dist.cdf)
    assert res_pvals.pvalue > 0.05, f'Null hypothesis rejected: generated random numbers are not uniform. Details: the (meta) p-value of the test of uniform distribution of p-values is {res_pvals.pvalue} which is not > 0.05'
    min_10pct_pval = np.percentile(ks_pvals, q=10)
    assert min_10pct_pval > 0.05, f'Null hypothesis rejected: generated random numbers are not uniform. Details: lower 10th quantile p-value of {min_10pct_pval} not > 0.05.'