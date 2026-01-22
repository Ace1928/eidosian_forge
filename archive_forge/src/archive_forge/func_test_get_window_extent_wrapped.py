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
def test_get_window_extent_wrapped():
    fig1 = plt.figure(figsize=(3, 3))
    fig1.suptitle('suptitle that is clearly too long in this case', wrap=True)
    window_extent_test = fig1._suptitle.get_window_extent()
    fig2 = plt.figure(figsize=(3, 3))
    fig2.suptitle('suptitle that is clearly\ntoo long in this case')
    window_extent_ref = fig2._suptitle.get_window_extent()
    assert window_extent_test.y0 == window_extent_ref.y0
    assert window_extent_test.y1 == window_extent_ref.y1