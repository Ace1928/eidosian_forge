import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.special as sc
@pytest.fixture
def log_softmax_expected():
    expected = np.array([-3.4401896985611953, -2.4401896985611953, -1.4401896985611953, -0.44018969856119533])
    return expected