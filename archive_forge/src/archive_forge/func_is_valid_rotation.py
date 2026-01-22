import math
import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
def is_valid_rotation(M):
    if not np.allclose(np.linalg.det(M), 1):
        return False
    return np.allclose(np.eye(3), np.dot(M, M.T))