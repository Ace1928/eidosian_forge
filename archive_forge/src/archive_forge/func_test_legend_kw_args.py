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
def test_legend_kw_args(self):
    fig, axs = plt.subplots(1, 2)
    lines = axs[0].plot(range(10))
    lines2 = axs[1].plot(np.arange(10) * 2.0)
    with mock.patch('matplotlib.legend.Legend') as Legend:
        fig.legend(loc='right', labels=('a', 'b'), handles=(lines, lines2))
    Legend.assert_called_with(fig, (lines, lines2), ('a', 'b'), loc='right', bbox_transform=fig.transFigure)