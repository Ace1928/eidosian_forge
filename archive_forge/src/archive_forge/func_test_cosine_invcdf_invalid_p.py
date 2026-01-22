import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special._ufuncs import _cosine_cdf, _cosine_invcdf
def test_cosine_invcdf_invalid_p():
    assert np.isnan(_cosine_invcdf([-0.1, 1.1])).all()