import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def test_ticket_1253(self):

    def linear(c, x):
        return c[0] * x + c[1]
    c = [2.0, 3.0]
    x = np.linspace(0, 10)
    y = linear(c, x)
    model = Model(linear)
    data = Data(x, y, wd=1.0, we=1.0)
    job = ODR(data, model, beta0=[1.0, 1.0])
    result = job.run()
    assert_equal(result.info, 2)