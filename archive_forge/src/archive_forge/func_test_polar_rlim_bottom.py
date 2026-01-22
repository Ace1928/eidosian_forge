import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@check_figures_equal(extensions=['png'])
def test_polar_rlim_bottom(fig_test, fig_ref):
    ax = fig_test.subplots(subplot_kw={'polar': True})
    ax.set_rlim(bottom=[0.5, 10])
    ax = fig_ref.subplots(subplot_kw={'polar': True})
    ax.set_rmax(10.0)
    ax.set_rmin(0.5)