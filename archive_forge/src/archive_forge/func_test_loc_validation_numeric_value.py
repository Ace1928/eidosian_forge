import collections
import platform
from unittest import mock
import warnings
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
import matplotlib.legend as mlegend
from matplotlib import _api, rc_context
from matplotlib.font_manager import FontProperties
def test_loc_validation_numeric_value():
    fig, ax = plt.subplots()
    ax.legend(loc=0)
    ax.legend(loc=1)
    ax.legend(loc=5)
    ax.legend(loc=10)
    with pytest.raises(ValueError, match='loc must be string, coordinate tuple, or an integer 0-10, not 11'):
        ax.legend(loc=11)
    with pytest.raises(ValueError, match='loc must be string, coordinate tuple, or an integer 0-10, not -1'):
        ax.legend(loc=-1)