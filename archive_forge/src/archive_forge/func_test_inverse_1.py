import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
@pytest.mark.parametrize('M, q', eg_pairs)
def test_inverse_1(M, q):
    iq = nq.inverse(q)
    iqM = nq.quat2mat(iq)
    iM = np.linalg.inv(M)
    assert np.allclose(iM, iqM)