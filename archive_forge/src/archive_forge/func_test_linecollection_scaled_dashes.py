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
def test_linecollection_scaled_dashes():
    lines1 = [[(0, 0.5), (0.5, 1)], [(0.3, 0.6), (0.2, 0.2)]]
    lines2 = [[[0.7, 0.2], [0.8, 0.4]], [[0.5, 0.7], [0.6, 0.1]]]
    lines3 = [[[0.6, 0.2], [0.8, 0.4]], [[0.5, 0.7], [0.1, 0.1]]]
    lc1 = mcollections.LineCollection(lines1, linestyles='--', lw=3)
    lc2 = mcollections.LineCollection(lines2, linestyles='-.')
    lc3 = mcollections.LineCollection(lines3, linestyles=':', lw=0.5)
    fig, ax = plt.subplots()
    ax.add_collection(lc1)
    ax.add_collection(lc2)
    ax.add_collection(lc3)
    leg = ax.legend([lc1, lc2, lc3], ['line1', 'line2', 'line 3'])
    h1, h2, h3 = leg.legend_handles
    for oh, lh in zip((lc1, lc2, lc3), (h1, h2, h3)):
        assert oh.get_linestyles()[0] == lh._dash_pattern