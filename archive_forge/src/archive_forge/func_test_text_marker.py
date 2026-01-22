import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.transforms import Affine2D
import pytest
@check_figures_equal(extensions=['png'], tol=1.86)
def test_text_marker(fig_ref, fig_test):
    ax_ref = fig_ref.add_subplot()
    ax_test = fig_test.add_subplot()
    ax_ref.plot(0, 0, marker='o', markersize=100, markeredgewidth=0)
    ax_test.plot(0, 0, marker='$\\bullet$', markersize=100, markeredgewidth=0)