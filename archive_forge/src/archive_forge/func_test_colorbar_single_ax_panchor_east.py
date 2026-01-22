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
@pytest.mark.parametrize('constrained', [False, True], ids=['standard', 'constrained'])
def test_colorbar_single_ax_panchor_east(constrained):
    fig = plt.figure(constrained_layout=constrained)
    ax = fig.add_subplot(111, anchor='N')
    plt.imshow([[0, 1]])
    plt.colorbar(panchor='E')
    assert ax.get_anchor() == 'E'