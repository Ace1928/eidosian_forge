import math
import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
def z_only(z):
    cosz = np.cos(z)
    sinz = np.sin(z)
    return np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])