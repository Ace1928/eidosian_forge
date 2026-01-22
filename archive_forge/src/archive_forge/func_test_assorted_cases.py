import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.special as sc
@pytest.mark.parametrize('a, b, x, expected', [(0.01, 150, -4, 0.9997368389767752), (1, 5, 0.01, 1.002003338101197), (50, 100, 0.01, 1.0050126452421464), (1, 0.3, -1000.0, -0.0007011932249442948), (1, 0.3, -10000.0, -7.001190321418937e-05), (9, 8.5, -350, -5.2240908319223784e-20), (9, 8.5, -355, -4.595407159813368e-20), (75, -123.5, 15, 3425753.920814889)])
def test_assorted_cases(self, a, b, x, expected):
    assert_allclose(sc.hyp1f1(a, b, x), expected, atol=0, rtol=1e-14)