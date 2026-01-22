from numpy.testing import assert_almost_equal
from statsmodels.sandbox.stats.runs import runstest_1samp
def test_median_cutoff():
    x = [1] * 5 + [2] * 6 + [3] * 8
    cutoff = 'median'
    expected = (-3.944254410803499, 8.004864125547193e-05)
    results = runstest_1samp(x, cutoff=cutoff, correction=False)
    assert_almost_equal(expected, results)