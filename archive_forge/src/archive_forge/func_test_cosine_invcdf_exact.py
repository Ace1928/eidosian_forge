import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special._ufuncs import _cosine_cdf, _cosine_invcdf
@pytest.mark.parametrize('p, expected', _cosinvcdf_exact)
def test_cosine_invcdf_exact(p, expected):
    assert _cosine_invcdf(p) == expected