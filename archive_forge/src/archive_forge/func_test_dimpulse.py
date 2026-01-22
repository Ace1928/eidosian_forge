import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
def test_dimpulse(self):
    a = np.asarray([[0.9, 0.1], [-0.2, 0.9]])
    b = np.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
    c = np.asarray([[0.1, 0.3]])
    d = np.asarray([[0.0, -0.1, 0.0]])
    dt = 0.5
    yout_imp_truth = (np.asarray([0.0, 0.04, 0.012, -0.0116, -0.03084, -0.045884, -0.056994, -0.06450548, -0.068804844, -0.0703091708]), np.asarray([-0.1, 0.025, 0.017, 0.00985, 0.00362, -0.0016595, -0.0059917, -0.009407675, -0.011960704, -0.01372089695]), np.asarray([0.0, -0.01, -0.003, 0.0029, 0.00771, 0.011471, 0.0142485, 0.01612637, 0.017201211, 0.0175772927]))
    tout, yout = dimpulse((a, b, c, d, dt), n=10)
    assert_equal(len(yout), 3)
    for i in range(0, len(yout)):
        assert_equal(yout[i].shape[0], 10)
        assert_array_almost_equal(yout[i].flatten(), yout_imp_truth[i])
    tfin = ([1.0], [1.0, 1.0], 0.5)
    yout_tfimpulse = np.asarray([0.0, 1.0, -1.0])
    tout, yout = dimpulse(tfin, n=3)
    assert_equal(len(yout), 1)
    assert_array_almost_equal(yout[0].flatten(), yout_tfimpulse)
    zpkin = tf2zpk(tfin[0], tfin[1]) + (0.5,)
    tout, yout = dimpulse(zpkin, n=3)
    assert_equal(len(yout), 1)
    assert_array_almost_equal(yout[0].flatten(), yout_tfimpulse)
    system = lti([1], [1, 1])
    assert_raises(AttributeError, dimpulse, system)