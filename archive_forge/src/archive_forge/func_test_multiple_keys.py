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
@image_comparison(['legend_multiple_keys.png'], remove_text=True)
def test_multiple_keys():
    fig, ax = plt.subplots()
    p1, = ax.plot([1, 2, 3], '-o')
    p2, = ax.plot([2, 3, 4], '-x')
    p3, = ax.plot([3, 4, 5], '-d')
    ax.legend([(p1, p2), (p2, p1), p3], ['two keys', 'pad=0', 'one key'], numpoints=1, handler_map={(p1, p2): HandlerTuple(ndivide=None), (p2, p1): HandlerTuple(ndivide=None, pad=0)})