import sys
import pytest
import numpy as np
from typing import NamedTuple
from numpy.testing import assert_allclose
from scipy.special import hyp2f1
from scipy.special._testutils import check_version, MissingModule
@pytest.mark.parametrize('hyp2f1_test_case', [pytest.param(Hyp2f1TestCase(a=-0.5, b=-0.9629749245209605, c=-15.5, z=1.1578947368421053 - 1.1578947368421053j, expected=0.9778506962676361 + 0.044083801141231616j, rtol=3e-12)), pytest.param(Hyp2f1TestCase(a=8.5, b=-3.9316537064827854, c=1.5, z=0.9473684210526314 - 0.10526315789473695j, expected=4.0793167523167675 - 10.11694246310966j, rtol=6e-12)), pytest.param(Hyp2f1TestCase(a=8.5, b=-0.9629749245209605, c=2.5, z=1.1578947368421053 - 0.10526315789473695j, expected=-2.9692999501916915 + 0.6394599899845594j, rtol=1e-11)), pytest.param(Hyp2f1TestCase(a=-0.5, b=-0.9629749245209605, c=-15.5, z=1.5789473684210522 - 1.1578947368421053j, expected=0.9493076367106102 - 0.04316852977183447j, rtol=1e-11)), pytest.param(Hyp2f1TestCase(a=-0.9220024191881196, b=-0.5, c=-15.5, z=0.5263157894736841 + 0.10526315789473673j, expected=0.9844377175631795 - 0.003120587561483841j, rtol=1e-10))])
def test_a_b_neg_int_after_euler_hypergeometric_transformation(self, hyp2f1_test_case):
    a, b, c, z, expected, rtol = hyp2f1_test_case
    assert abs(c - a - int(c - a)) < 1e-15 and c - a < 0 or (abs(c - b - int(c - b)) < 1e-15 and c - b < 0)
    assert_allclose(hyp2f1(a, b, c, z), expected, rtol=rtol)