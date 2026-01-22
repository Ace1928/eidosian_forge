from datetime import datetime
import io
import warnings
import numpy as np
from numpy.testing import assert_almost_equal
from packaging.version import parse as parse_version
import pyparsing
import pytest
import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib.text import Text, Annotation, OffsetFrom
@check_figures_equal(extensions=['png'])
def test_annotate_and_offsetfrom_copy_input(fig_test, fig_ref):
    ax = fig_test.add_subplot()
    l, = ax.plot([0, 2], [0, 2])
    of_xy = np.array([0.5, 0.5])
    ax.annotate('foo', textcoords=OffsetFrom(l, of_xy), xytext=(10, 0), xy=(0, 0))
    of_xy[:] = 1
    ax = fig_ref.add_subplot()
    l, = ax.plot([0, 2], [0, 2])
    an_xy = np.array([0.5, 0.5])
    ax.annotate('foo', xy=an_xy, xycoords=l, xytext=(10, 0), textcoords='offset points')
    an_xy[:] = 2