import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.stats._tukeylambda_stats import (tukeylambda_variance,
def test_tukeylambda_stats_mpmath():
    """Compare results with some values that were computed using mpmath."""
    a10 = dict(atol=1e-10, rtol=0)
    a12 = dict(atol=1e-12, rtol=0)
    data = [[-0.1, 4.780502178742536, 3.785595203464545], [-0.0649, 4.164280235998958, 2.520196759474357], [-0.05, 3.9367226789077527, 2.1312979305777726], [-0.001, 3.301283803909649, 1.2145246008354298], [0.001, 3.278507756495722, 1.1856063477928758], [0.03125, 2.959278032546158, 0.80448755516182], [0.05, 2.782810534054645, 0.6116040438866444], [0.0649, 2.6528238675410054, 0.47683411953277455], [1.2, 0.24215392057858834, -1.2342804716904971], [10.0, 0.000952375797577036, 2.3781069735514495], [20.0, 0.00012195121951131043, 7.376543210027095]]
    for lam, var_expected, kurt_expected in data:
        var = tukeylambda_variance(lam)
        assert_allclose(var, var_expected, **a12)
        kurt = tukeylambda_kurtosis(lam)
        assert_allclose(kurt, kurt_expected, **a10)
    lam, var_expected, kurt_expected = zip(*data)
    var = tukeylambda_variance(lam)
    assert_allclose(var, var_expected, **a12)
    kurt = tukeylambda_kurtosis(lam)
    assert_allclose(kurt, kurt_expected, **a10)