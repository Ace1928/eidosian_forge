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
@pytest.mark.parametrize('label_array', [['low', 'high'], ('low', 'high'), np.array(['low', 'high'])])
def test_plot_single_input_multiple_label(label_array):
    x = [1, 2, 3]
    y = [2, 5, 6]
    fig, ax = plt.subplots()
    ax.plot(x, y, label=label_array)
    leg = ax.legend()
    assert len(leg.get_texts()) == 1
    assert leg.get_texts()[0].get_text() == str(label_array)