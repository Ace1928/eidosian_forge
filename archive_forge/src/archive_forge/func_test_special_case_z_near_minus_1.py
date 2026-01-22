import sys
import pytest
import numpy as np
from typing import NamedTuple
from numpy.testing import assert_allclose
from scipy.special import hyp2f1
from scipy.special._testutils import check_version, MissingModule
@pytest.mark.parametrize('hyp2f1_test_case', [pytest.param(Hyp2f1TestCase(a=0.5, b=0.2, c=1.3, z=-1 + 0j, expected=0.9428846409614143 + 0j, rtol=1e-15)), pytest.param(Hyp2f1TestCase(a=12.3, b=8.0, c=5.300000000000001, z=-1 + 0j, expected=-4.845809986595704e-06 + 0j, rtol=1e-15)), pytest.param(Hyp2f1TestCase(a=221.5, b=90.2, c=132.3, z=-1 + 0j, expected=2.0490488728377282e-42 + 0j, rtol=1e-07)), pytest.param(Hyp2f1TestCase(a=-102.1, b=-20.3, c=-80.8, z=-1 + 0j, expected=45143784.46783885 + 0j, rtol=1e-07), marks=pytest.mark.xfail(condition=sys.maxsize < 2 ** 32, reason='Fails on 32 bit.'))])
def test_special_case_z_near_minus_1(self, hyp2f1_test_case):
    """Tests for case z ~ -1, c ~ 1 + a - b

        Expected answers computed using mpmath.
        """
    a, b, c, z, expected, rtol = hyp2f1_test_case
    assert abs(1 + a - b - c) < 1e-15 and abs(z + 1) < 1e-15
    assert_allclose(hyp2f1(a, b, c, z), expected, rtol=rtol)