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
def test_setting_alpha_keeps_polycollection_color():
    pc = plt.fill_between([0, 1], [2, 3], color='#123456', label='label')
    patch = plt.legend().get_patches()[0]
    patch.set_alpha(0.5)
    assert patch.get_facecolor()[:3] == tuple(pc.get_facecolor()[0][:3])
    assert patch.get_edgecolor()[:3] == tuple(pc.get_edgecolor()[0][:3])