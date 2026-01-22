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
def test_figure_legend_outside():
    todos = ['upper ' + pos for pos in ['left', 'center', 'right']]
    todos += ['lower ' + pos for pos in ['left', 'center', 'right']]
    todos += ['left ' + pos for pos in ['lower', 'center', 'upper']]
    todos += ['right ' + pos for pos in ['lower', 'center', 'upper']]
    upperext = [20.347556, 27.722556, 790.583, 545.499]
    lowerext = [20.347556, 71.056556, 790.583, 588.833]
    leftext = [151.681556, 27.722556, 790.583, 588.833]
    rightext = [20.347556, 27.722556, 659.249, 588.833]
    axbb = [upperext, upperext, upperext, lowerext, lowerext, lowerext, leftext, leftext, leftext, rightext, rightext, rightext]
    legbb = [[10.0, 555.0, 133.0, 590.0], [338.5, 555.0, 461.5, 590.0], [667, 555.0, 790.0, 590.0], [10.0, 10.0, 133.0, 45.0], [338.5, 10.0, 461.5, 45.0], [667.0, 10.0, 790.0, 45.0], [10.0, 10.0, 133.0, 45.0], [10.0, 282.5, 133.0, 317.5], [10.0, 555.0, 133.0, 590.0], [667, 10.0, 790.0, 45.0], [667.0, 282.5, 790.0, 317.5], [667.0, 555.0, 790.0, 590.0]]
    for nn, todo in enumerate(todos):
        print(todo)
        fig, axs = plt.subplots(constrained_layout=True, dpi=100)
        axs.plot(range(10), label='Boo1')
        leg = fig.legend(loc='outside ' + todo)
        fig.draw_without_rendering()
        assert_allclose(axs.get_window_extent().extents, axbb[nn])
        assert_allclose(leg.get_window_extent().extents, legbb[nn])