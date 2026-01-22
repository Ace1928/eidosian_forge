import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
import scipy.interpolate.interpnd as interpnd
import scipy.spatial._qhull as qhull
import pickle
def ip(x, y):
    t1 = x + y <= 1
    t2 = ~t1
    x1 = x[t1]
    y1 = y[t1]
    x2 = x[t2]
    y2 = y[t2]
    z = 0 * x
    z[t1] = values[0] * (1 - x1 - y1) + values[1] * y1 + values[3] * x1
    z[t2] = values[2] * (x2 + y2 - 1) + values[1] * (1 - x2) + values[3] * (1 - y2)
    return z