import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_tripcolor_clim():
    np.random.seed(19680801)
    a, b, c = (np.random.rand(10), np.random.rand(10), np.random.rand(10))
    ax = plt.figure().add_subplot()
    clim = (0.25, 0.75)
    norm = ax.tripcolor(a, b, c, clim=clim).norm
    assert (norm.vmin, norm.vmax) == clim