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
def test_radio_buttons_activecolor_change(fig_test, fig_ref):
    widgets.RadioButtons(fig_ref.subplots(), ['tea', 'coffee'], activecolor='green')
    cb = widgets.RadioButtons(fig_test.subplots(), ['tea', 'coffee'], activecolor='red')
    cb.activecolor = 'green'