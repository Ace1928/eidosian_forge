import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from statsmodels.datasets import elnino
from statsmodels.graphics.functional import (
@pytest.mark.matplotlib
def test_fboxplot_rainbowplot(close_figures):

    def harmfunc(t):
        """Test function, combination of a few harmonic terms."""
        ci = int(np.random.random() > 0.9)
        a1i = np.random.random() * 0.05
        a2i = np.random.random() * 0.05
        b1i = (0.15 - 0.1) * np.random.random() + 0.1
        b2i = (0.15 - 0.1) * np.random.random() + 0.1
        func = (1 - ci) * (a1i * np.sin(t) + a2i * np.cos(t)) + ci * (b1i * np.sin(t) + b2i * np.cos(t))
        return func
    np.random.seed(1234567)
    t = np.linspace(0, 2 * np.pi, 250)
    data = [harmfunc(t) for _ in range(20)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _, depth, ix_depth, ix_outliers = fboxplot(data, wfactor=2, ax=ax)
    ix_expected = np.array([13, 4, 15, 19, 8, 6, 3, 16, 9, 7, 1, 5, 2, 12, 17, 11, 14, 10, 0, 18])
    assert_equal(ix_depth, ix_expected)
    ix_expected2 = np.array([2, 11, 17, 18])
    assert_equal(ix_outliers, ix_expected2)
    xdata = np.arange(data[0].size)
    fig = rainbowplot(data, xdata=xdata, depth=depth, cmap=plt.cm.rainbow)