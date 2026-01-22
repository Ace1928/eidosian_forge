import functools
import io
from unittest import mock
import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
import matplotlib.colors as mcolors
import matplotlib.widgets as widgets
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing.widgets import (click_and_drag, do_event, get_ax,
import numpy as np
from numpy.testing import assert_allclose
import pytest
def test_deprecation_selector_visible_attribute(ax):
    tool = widgets.RectangleSelector(ax, lambda *args: None)
    assert tool.get_visible()
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match='was deprecated in Matplotlib 3.8'):
        tool.visible