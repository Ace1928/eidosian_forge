import sys
import pytest
import numpy as np
from typing import NamedTuple
from numpy.testing import assert_allclose
from scipy.special import hyp2f1
from scipy.special._testutils import check_version, MissingModule
@pytest.mark.parametrize('hyp2f1_test_case', [pytest.param(Hyp2f1TestCase(a=0.5, b=0.2, c=1.5, z=1 + 0j, expected=1.1496439092239847 + 0j, rtol=1e-15)), pytest.param(Hyp2f1TestCase(a=12.3, b=8.0, c=20.31, z=1 + 0j, expected=69280986.75273195 + 0j, rtol=1e-15)), pytest.param(Hyp2f1TestCase(a=290.2, b=321.5, c=700.1, z=1 + 0j, expected=1.3396562400934e+117 + 0j, rtol=1e-12)), pytest.param(Hyp2f1TestCase(a=9.2, b=621.5, c=700.1, z=1 + 0j, expected=952726652.4158565 + 0j, rtol=5e-13)), pytest.param(Hyp2f1TestCase(a=621.5, b=9.2, c=700.1, z=1 + 0j, expected=952726652.4160284 + 0j, rtol=5e-12)), pytest.param(Hyp2f1TestCase(a=-101.2, b=-400.4, c=-172.1, z=1 + 0j, expected=2.2253618341394838e+37 + 0j, rtol=1e-13)), pytest.param(Hyp2f1TestCase(a=-400.4, b=-101.2, c=-172.1, z=1 + 0j, expected=2.2253618341394838e+37 + 0j, rtol=5e-13)), pytest.param(Hyp2f1TestCase(a=172.5, b=-201.3, c=151.2, z=1 + 0j, expected=7.072266653650905e-135 + 0j, rtol=5e-13)), pytest.param(Hyp2f1TestCase(a=-201.3, b=172.5, c=151.2, z=1 + 0j, expected=7.072266653650905e-135 + 0j, rtol=5e-13)), pytest.param(Hyp2f1TestCase(a=-102.1, b=-20.3, c=1.3, z=1 + 0j, expected=2.7899070752746906e+22 + 0j, rtol=3e-14)), pytest.param(Hyp2f1TestCase(a=-202.6, b=60.3, c=1.5, z=1 + 0j, expected=-1.3113641413099326e-56 + 0j, rtol=1e-12))])
def test_unital_argument(self, hyp2f1_test_case):
    """Tests for case z = 1, c - a - b > 0.

        Expected answers computed using mpmath.
        """
    a, b, c, z, expected, rtol = hyp2f1_test_case
    assert z == 1 and c - a - b > 0
    assert_allclose(hyp2f1(a, b, c, z), expected, rtol=rtol)