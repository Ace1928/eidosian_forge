import copy
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from matplotlib import scale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_pcolormesh_gouraud_nans():
    np.random.seed(19680801)
    values = np.linspace(0, 180, 3)
    radii = np.linspace(100, 1000, 10)
    z, y = np.meshgrid(values, radii)
    x = np.radians(np.random.rand(*z.shape) * 100)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.set_rlim(101, 1000)
    ax.pcolormesh(x, y, z, shading='gouraud')
    fig.canvas.draw()