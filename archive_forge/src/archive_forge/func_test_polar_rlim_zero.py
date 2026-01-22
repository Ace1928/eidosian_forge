import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_polar_rlim_zero():
    ax = plt.figure().add_subplot(projection='polar')
    ax.plot(np.arange(10), np.arange(10) + 0.01)
    assert ax.get_ylim()[0] == 0