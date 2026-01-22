import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
@image_comparison(['tight_layout6'])
def test_tight_layout6():
    """Test tight_layout for gridspec."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fig = plt.figure()
        gs1 = mpl.gridspec.GridSpec(2, 1)
        ax1 = fig.add_subplot(gs1[0])
        ax2 = fig.add_subplot(gs1[1])
        example_plot(ax1)
        example_plot(ax2)
        gs1.tight_layout(fig, rect=[0, 0, 0.5, 1])
        gs2 = mpl.gridspec.GridSpec(3, 1)
        for ss in gs2:
            ax = fig.add_subplot(ss)
            example_plot(ax)
            ax.set_title('')
            ax.set_xlabel('')
        ax.set_xlabel('x-label', fontsize=12)
        gs2.tight_layout(fig, rect=[0.5, 0, 1, 1], h_pad=0.45)
        top = min(gs1.top, gs2.top)
        bottom = max(gs1.bottom, gs2.bottom)
        gs1.tight_layout(fig, rect=[None, 0 + (bottom - gs1.bottom), 0.5, 1 - (gs1.top - top)])
        gs2.tight_layout(fig, rect=[0.5, 0 + (bottom - gs2.bottom), None, 1 - (gs2.top - top)], h_pad=0.45)