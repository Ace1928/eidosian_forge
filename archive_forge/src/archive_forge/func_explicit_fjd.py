import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def explicit_fjd(self, B, x):
    eBx = np.exp(B[2] * x)
    ret = B[1] * 2.0 * (eBx - 1.0) * B[2] * eBx
    return ret