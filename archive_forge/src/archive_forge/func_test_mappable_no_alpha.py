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
def test_mappable_no_alpha():
    fig, ax = plt.subplots()
    sm = cm.ScalarMappable(norm=mcolors.Normalize(), cmap='viridis')
    fig.colorbar(sm, ax=ax)
    sm.set_cmap('plasma')
    plt.draw()