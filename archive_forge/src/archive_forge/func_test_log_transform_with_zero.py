import re
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.backend_bases import MouseEvent
@image_comparison(['semi_log_with_zero.png'], style='mpl20')
def test_log_transform_with_zero():
    x = np.arange(-10, 10)
    y = (1.0 - 1.0 / (x ** 2 + 1)) ** 20
    fig, ax = plt.subplots()
    ax.semilogy(x, y, '-o', lw=15, markeredgecolor='k')
    ax.set_ylim(1e-07, 1)
    ax.grid(True)