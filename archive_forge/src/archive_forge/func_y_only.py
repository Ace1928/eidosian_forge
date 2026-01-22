import math
import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
def y_only(y):
    cosy = np.cos(y)
    siny = np.sin(y)
    return np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])