import math
import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
def test_euler_mat_1():
    M = nea.euler2mat()
    assert_array_equal(M, np.eye(3))