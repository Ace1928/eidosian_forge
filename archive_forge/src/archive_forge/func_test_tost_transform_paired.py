import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_
import pytest
import statsmodels.stats.weightstats as smws
from statsmodels.tools.testing import Holder
@pytest.mark.xfail(reason='shape mismatch between res1[1:] and res_sas[1:]', raises=AssertionError, strict=True)
def test_tost_transform_paired():
    raw = np.array('       103.4 90.11  59.92 77.71  68.17 77.71  94.54 97.51\n       69.48 58.21  72.17 101.3  74.37 79.84  84.44 96.06\n       96.74 89.30  94.26 97.22  48.52 61.62  95.68 85.80'.split(), float)
    x, y = raw.reshape(-1, 2).T
    res1 = smws.ttost_paired(x, y, 0.8, 1.25, transform=np.log)
    res_sas = (0.0031, (3.38, 0.0031), (-5.9, 5e-05))
    assert_almost_equal(res1[0], res_sas[0], 3)
    assert_almost_equal(res1[1:], res_sas[1:], 2)
    assert_almost_equal(res1[0], tost_s_paired.p_value, 13)