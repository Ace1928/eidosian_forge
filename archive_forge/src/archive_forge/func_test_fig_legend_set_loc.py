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
@pytest.mark.parametrize('loc', ('outside right', 'right'))
def test_fig_legend_set_loc(loc):
    fig, ax = plt.subplots()
    ax.plot(range(10), label='test')
    leg = fig.legend()
    leg.set_loc(loc)
    loc = loc.split()[1] if loc.startswith('outside') else loc
    assert leg._get_loc() == mlegend.Legend.codes[loc]