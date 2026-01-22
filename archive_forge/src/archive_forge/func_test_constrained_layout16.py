import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@image_comparison(['constrained_layout16.png'])
def test_constrained_layout16():
    """Test ax.set_position."""
    fig, ax = plt.subplots(layout='constrained')
    example_plot(ax, fontsize=12)
    ax2 = fig.add_axes([0.2, 0.2, 0.4, 0.4])