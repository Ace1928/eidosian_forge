import warnings # for silencing, see above...
import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats, special
from statsmodels.sandbox.distributions.extras import (
def test_equivalent_negsq(self):
    xx, nxx, ppfq = (self.xx, self.nxx, self.ppfq)
    d1, d2 = (negsquarenormalg, stats.chi2(1))
    assert_almost_equal(d1.cdf(nxx), 1 - d2.cdf(xx), err_msg='cdf' + d1.name)
    assert_almost_equal(d1.pdf(nxx), d2.pdf(xx))
    assert_almost_equal(d1.sf(nxx), 1 - d2.sf(xx))
    assert_almost_equal(d1.ppf(ppfq), -d2.ppf(ppfq)[::-1])
    assert_almost_equal(d1.isf(ppfq), -d2.isf(ppfq)[::-1])
    assert_almost_equal(d1.moment(3), -d2.moment(3))
    ch2oddneg = [v * (-1) ** (i + 1) for i, v in enumerate(d2.stats(moments='mvsk'))]
    assert_almost_equal(d1.stats(moments='mvsk'), ch2oddneg, err_msg='stats ' + d1.name + d2.name)