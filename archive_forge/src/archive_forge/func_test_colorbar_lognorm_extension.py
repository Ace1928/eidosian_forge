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
@pytest.mark.parametrize('extend', ['both', 'min', 'max'])
def test_colorbar_lognorm_extension(extend):
    f, ax = plt.subplots()
    cb = Colorbar(ax, norm=LogNorm(vmin=0.1, vmax=1000.0), orientation='vertical', extend=extend)
    assert cb._values[0] >= 0.0