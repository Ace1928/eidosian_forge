import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
@image_comparison(['tight_layout_offsetboxes1', 'tight_layout_offsetboxes2'])
def test_tight_layout_offsetboxes():
    rows = cols = 2
    colors = ['red', 'blue', 'green', 'yellow']
    x = y = [0, 1]

    def _subplots():
        _, axs = plt.subplots(rows, cols)
        axs = axs.flat
        for ax, color in zip(axs, colors):
            ax.plot(x, y, color=color)
            add_offsetboxes(ax, 20, color=color)
        return axs
    axs = _subplots()
    plt.tight_layout()
    axs = _subplots()
    for ax in axs[cols - 1::rows]:
        for child in ax.get_children():
            if isinstance(child, AnchoredOffsetbox):
                child.set_visible(False)
    plt.tight_layout()