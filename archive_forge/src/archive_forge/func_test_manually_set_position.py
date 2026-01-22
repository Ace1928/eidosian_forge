import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
def test_manually_set_position():
    fig, axs = plt.subplots(1, 2, layout='constrained')
    axs[0].set_position([0.2, 0.2, 0.3, 0.3])
    fig.draw_without_rendering()
    pp = axs[0].get_position()
    np.testing.assert_allclose(pp, [[0.2, 0.2], [0.5, 0.5]])
    fig, axs = plt.subplots(1, 2, layout='constrained')
    axs[0].set_position([0.2, 0.2, 0.3, 0.3])
    pc = axs[0].pcolormesh(np.random.rand(20, 20))
    fig.colorbar(pc, ax=axs[0])
    fig.draw_without_rendering()
    pp = axs[0].get_position()
    np.testing.assert_allclose(pp, [[0.2, 0.2], [0.44, 0.5]])