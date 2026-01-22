import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
@pytest.mark.parametrize('vec', np.eye(3))
@pytest.mark.parametrize('M, q', eg_pairs)
def test_qrotate(vec, M, q):
    vdash = nq.rotate_vector(vec, q)
    vM = M @ vec
    assert_array_almost_equal(vdash, vM)