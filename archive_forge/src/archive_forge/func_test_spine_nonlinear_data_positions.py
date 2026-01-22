import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.spines import Spines
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@check_figures_equal(extensions=['png'])
def test_spine_nonlinear_data_positions(fig_test, fig_ref):
    plt.style.use('default')
    ax = fig_test.add_subplot()
    ax.set(xscale='log', xlim=(0.1, 1))
    ax.spines.left.set_position(('data', 1))
    ax.spines.left.set_linewidth(2)
    ax.spines.right.set_position(('data', 0.1))
    ax.tick_params(axis='y', labelleft=False, direction='in')
    ax = fig_ref.add_subplot()
    ax.set(xscale='log', xlim=(0.1, 1))
    ax.spines.right.set_linewidth(2)
    ax.tick_params(axis='y', labelleft=False, left=False, right=True)