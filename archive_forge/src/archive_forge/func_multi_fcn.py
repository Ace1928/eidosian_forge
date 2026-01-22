import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def multi_fcn(self, B, x):
    if (x < 0.0).any():
        raise OdrStop
    theta = pi * B[3] / 2.0
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    omega = np.power(2.0 * pi * x * np.exp(-B[2]), B[3])
    phi = np.arctan2(omega * stheta, 1.0 + omega * ctheta)
    r = (B[0] - B[1]) * np.power(np.sqrt(np.power(1.0 + omega * ctheta, 2) + np.power(omega * stheta, 2)), -B[4])
    ret = np.vstack([B[1] + r * np.cos(B[4] * phi), r * np.sin(B[4] * phi)])
    return ret