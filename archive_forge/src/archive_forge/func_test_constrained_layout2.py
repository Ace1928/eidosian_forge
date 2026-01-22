import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@image_comparison(['constrained_layout2.png'])
def test_constrained_layout2():
    """Test constrained_layout for 2x2 subplots"""
    fig, axs = plt.subplots(2, 2, layout='constrained')
    for ax in axs.flat:
        example_plot(ax, fontsize=24)