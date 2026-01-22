import platform
import numpy as np
import pytest
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib import rc_context
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.colors import (
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import FixedLocator, LogFormatter, StrMethodFormatter
from matplotlib.testing.decorators import check_figures_equal
def test_colorbar_get_ticks_2():
    plt.rcParams['_internal.classic_mode'] = False
    fig, ax = plt.subplots()
    pc = ax.pcolormesh([[0.05, 0.95]])
    cb = fig.colorbar(pc)
    np.testing.assert_allclose(cb.get_ticks(), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])