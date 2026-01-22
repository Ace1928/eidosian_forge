import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
def test_constrained_layout19():
    """Test twiny"""
    fig, ax = plt.subplots(layout='constrained')
    ax2 = ax.twiny()
    example_plot(ax)
    example_plot(ax2, fontsize=24)
    ax2.set_title('')
    ax.set_title('')
    fig.draw_without_rendering()
    assert all(ax.get_position().extents == ax2.get_position().extents)