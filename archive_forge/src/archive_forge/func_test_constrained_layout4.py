import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@image_comparison(['constrained_layout4.png'])
def test_constrained_layout4():
    """Test constrained_layout for a single colorbar with subplots"""
    fig, axs = plt.subplots(2, 2, layout='constrained')
    for ax in axs.flat:
        pcm = example_pcolor(ax, fontsize=24)
    fig.colorbar(pcm, ax=axs, pad=0.01, shrink=0.6)