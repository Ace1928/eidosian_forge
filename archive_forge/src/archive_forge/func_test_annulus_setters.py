import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.patches import (Annulus, Ellipse, Patch, Polygon, Rectangle,
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib import (
import sys
@image_comparison(baseline_images=['annulus'], extensions=['png'])
def test_annulus_setters():
    fig, ax = plt.subplots()
    cir = Annulus((0.0, 0.0), 0.2, 0.01, fc='g')
    ell = Annulus((0.0, 0.0), (1, 2), 0.1, 0, fc='m', ec='b', alpha=0.5, hatch='xxx')
    ax.add_patch(cir)
    ax.add_patch(ell)
    ax.set_aspect('equal')
    cir.center = (0.5, 0.5)
    cir.radii = 0.2
    cir.width = 0.05
    ell.center = (0.5, 0.5)
    ell.radii = (0.5, 0.3)
    ell.width = 0.1
    ell.angle = 45