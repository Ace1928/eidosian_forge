import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
def test_constrained_layout20():
    """Smoke test cl does not mess up added axes"""
    gx = np.linspace(-5, 5, 4)
    img = np.hypot(gx, gx[:, None])
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    mesh = ax.pcolormesh(gx, gx, img[:-1, :-1])
    fig.colorbar(mesh)