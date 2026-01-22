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
@check_figures_equal(extensions=['png'])
def test_check_button_props(fig_test, fig_ref):
    label_props = {'color': ['red'], 'fontsize': [24]}
    frame_props = {'facecolor': 'green', 'edgecolor': 'blue', 'linewidth': 2}
    check_props = {'facecolor': 'red', 'linewidth': 2}
    widgets.CheckButtons(fig_ref.subplots(), ['tea', 'coffee'], [True, True], label_props=label_props, frame_props=frame_props, check_props=check_props)
    cb = widgets.CheckButtons(fig_test.subplots(), ['tea', 'coffee'], [True, True])
    cb.set_label_props(label_props)
    cb.set_frame_props({**frame_props, 's': (24 / 2) ** 2})
    check_props['edgecolor'] = check_props.pop('facecolor')
    cb.set_check_props({**check_props, 's': (24 / 2) ** 2})