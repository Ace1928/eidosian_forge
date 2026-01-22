import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_polar_no_data():
    plt.subplot(projection='polar')
    ax = plt.gca()
    assert ax.get_rmin() == 0 and ax.get_rmax() == 1
    plt.close('all')
    plt.polar()
    ax = plt.gca()
    assert ax.get_rmin() == 0 and ax.get_rmax() == 1