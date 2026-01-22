import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def test_ticket_11800(self):
    beta_true = np.array([1.0, 2.3, 1.1, -1.0, 1.3, 0.5])
    nr_measurements = 10
    std_dev_x = 0.01
    x_error = np.array([[0.00063445, 0.00515731, 0.00162719, 0.01022866, -0.01624845, 0.00482652, 0.00275988, -0.00714734, -0.00929201, -0.00687301], [-0.00831623, -0.00821211, -0.00203459, 0.00938266, -0.00701829, 0.0032169, 0.00259194, -0.00581017, -0.0030283, 0.01014164]])
    std_dev_y = 0.05
    y_error = np.array([[0.05275304, 0.04519563, -0.07524086, 0.03575642, 0.04745194, 0.03806645, 0.07061601, -0.00753604, -0.02592543, -0.02394929], [0.03632366, 0.06642266, 0.08373122, 0.03988822, -0.0092536, -0.03750469, -0.03198903, 0.01642066, 0.01293648, -0.05627085]])
    beta_solution = np.array([2.6292023575666588, -126.60848499629961, 129.70357277540307, -1.8856098540118547, 78.38341607712749, -76.41240768380871])

    def func(beta, x):
        y0 = beta[0] + beta[1] * x[0, :] + beta[2] * x[1, :]
        y1 = beta[3] + beta[4] * x[0, :] + beta[5] * x[1, :]
        return np.vstack((y0, y1))

    def df_dbeta_odr(beta, x):
        nr_meas = np.shape(x)[1]
        zeros = np.zeros(nr_meas)
        ones = np.ones(nr_meas)
        dy0 = np.array([ones, x[0, :], x[1, :], zeros, zeros, zeros])
        dy1 = np.array([zeros, zeros, zeros, ones, x[0, :], x[1, :]])
        return np.stack((dy0, dy1))

    def df_dx_odr(beta, x):
        nr_meas = np.shape(x)[1]
        ones = np.ones(nr_meas)
        dy0 = np.array([beta[1] * ones, beta[2] * ones])
        dy1 = np.array([beta[4] * ones, beta[5] * ones])
        return np.stack((dy0, dy1))
    x0_true = np.linspace(1, 10, nr_measurements)
    x1_true = np.linspace(1, 10, nr_measurements)
    x_true = np.array([x0_true, x1_true])
    y_true = func(beta_true, x_true)
    x_meas = x_true + x_error
    y_meas = y_true + y_error
    model_f = Model(func, fjacb=df_dbeta_odr, fjacd=df_dx_odr)
    data = RealData(x_meas, y_meas, sx=std_dev_x, sy=std_dev_y)
    odr_obj = ODR(data, model_f, beta0=0.9 * beta_true, maxit=100)
    odr_obj.set_job(deriv=3)
    odr_out = odr_obj.run()
    assert_equal(odr_out.info, 1)
    assert_array_almost_equal(odr_out.beta, beta_solution)