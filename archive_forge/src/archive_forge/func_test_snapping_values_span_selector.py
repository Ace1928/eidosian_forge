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
def test_snapping_values_span_selector(ax):

    def onselect(*args):
        pass
    tool = widgets.SpanSelector(ax, onselect, direction='horizontal')
    snap_function = tool._snap
    snap_values = np.linspace(0, 5, 11)
    values = np.array([-0.1, 0.1, 0.2, 0.5, 0.6, 0.7, 0.9, 4.76, 5.0, 5.5])
    expect = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 5.0, 5.0, 5.0])
    values = snap_function(values, snap_values)
    assert_allclose(values, expect)