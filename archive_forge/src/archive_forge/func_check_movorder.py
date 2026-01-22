import numpy as np
from scipy import signal
from numpy.testing import assert_array_equal, assert_array_almost_equal
def check_movorder():
    """graphical test for movorder"""
    import matplotlib.pylab as plt
    x = np.arange(1, 10)
    xo = movorder(x, order='max')
    assert_array_equal(xo, x)
    x = np.arange(10, 1, -1)
    xo = movorder(x, order='min')
    assert_array_equal(xo, x)
    assert_array_equal(movorder(x, order='min', lag='centered')[:-1], x[1:])
    tt = np.linspace(0, 2 * np.pi, 15)
    x = np.sin(tt) + 1
    xo = movorder(x, order='max')
    plt.figure()
    plt.plot(tt, x, '.-', tt, xo, '.-')
    plt.title('moving max lagged')
    xo = movorder(x, order='max', lag='centered')
    plt.figure()
    plt.plot(tt, x, '.-', tt, xo, '.-')
    plt.title('moving max centered')
    xo = movorder(x, order='max', lag='leading')
    plt.figure()
    plt.plot(tt, x, '.-', tt, xo, '.-')
    plt.title('moving max leading')