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
@check_figures_equal(extensions=['png'])
def test_remove_cb_whose_mappable_has_no_figure(fig_ref, fig_test):
    ax = fig_test.add_subplot()
    cb = fig_test.colorbar(cm.ScalarMappable(), cax=ax)
    cb.remove()