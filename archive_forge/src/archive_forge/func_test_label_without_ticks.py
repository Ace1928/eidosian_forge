import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.spines import Spines
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_label_without_ticks():
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.3, bottom=0.3)
    ax.plot(np.arange(10))
    ax.yaxis.set_ticks_position('left')
    ax.spines.left.set_position(('outward', 30))
    ax.spines.right.set_visible(False)
    ax.set_ylabel('y label')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines.bottom.set_position(('outward', 30))
    ax.spines.top.set_visible(False)
    ax.set_xlabel('x label')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    plt.draw()
    spine = ax.spines.left
    spinebbox = spine.get_transform().transform_path(spine.get_path()).get_extents()
    assert ax.yaxis.label.get_position()[0] < spinebbox.xmin, 'Y-Axis label not left of the spine'
    spine = ax.spines.bottom
    spinebbox = spine.get_transform().transform_path(spine.get_path()).get_extents()
    assert ax.xaxis.label.get_position()[1] < spinebbox.ymin, 'X-Axis label not below the spine'