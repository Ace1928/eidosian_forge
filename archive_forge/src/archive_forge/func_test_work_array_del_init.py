import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def test_work_array_del_init(self):
    """
        Verify fix for gh-18739 where del_init=1 fails.
        """

    def func(b, x):
        return b[0] + b[1] * x
    n_data = 4
    x = np.arange(n_data)
    y = np.where(x % 2, x + 0.1, x - 0.1)
    x_err = np.full(n_data, 0.1)
    y_err = np.full(n_data, 0.1)
    linear_model = Model(func)
    rd0 = RealData(x, y, sx=x_err, sy=y_err)
    rd1 = RealData(x, y, sx=x_err, sy=0.1)
    rd2 = RealData(x, y, sx=x_err, sy=[0.1])
    rd3 = RealData(x, y, sx=x_err, sy=np.full((1, n_data), 0.1))
    rd4 = RealData(x, y, sx=x_err, covy=[[0.01]])
    rd5 = RealData(x, y, sx=x_err, covy=np.full((1, 1, n_data), 0.01))
    for rd in [rd0, rd1, rd2, rd3, rd4, rd5]:
        odr_obj = ODR(rd, linear_model, beta0=[0.4, 0.4], delta0=np.full(n_data, -0.1))
        odr_obj.set_job(fit_type=0, del_init=1)
        odr_obj.run()