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
def test_offset_text_loc():
    plt.style.use('mpl20')
    fig, ax = plt.subplots()
    np.random.seed(seed=19680808)
    pc = ax.pcolormesh(np.random.randn(10, 10) * 1000000.0)
    cb = fig.colorbar(pc, location='right', extend='max')
    fig.draw_without_rendering()
    assert cb.ax.yaxis.offsetText.get_position()[1] > ax.bbox.y1