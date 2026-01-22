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
@check_figures_equal(extensions=['png', 'svg', 'pdf', 'eps'])
def test_modifying_arc(fig_test, fig_ref):
    arc1 = Arc([0.5, 0.5], 0.5, 1, theta1=0, theta2=60, angle=20)
    arc2 = Arc([0.5, 0.5], 1.5, 1, theta1=0, theta2=60, angle=10)
    fig_ref.subplots().add_patch(arc1)
    fig_test.subplots().add_patch(arc2)
    arc2.set_width(0.5)
    arc2.set_angle(20)