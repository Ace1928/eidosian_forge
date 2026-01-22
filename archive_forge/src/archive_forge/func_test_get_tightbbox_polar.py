import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_get_tightbbox_polar():
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig.canvas.draw()
    bb = ax.get_tightbbox(fig.canvas.get_renderer())
    assert_allclose(bb.extents, [107.7778, 29.2778, 539.7847, 450.7222], rtol=0.001)