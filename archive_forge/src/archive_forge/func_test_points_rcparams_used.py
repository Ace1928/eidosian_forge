import matplotlib.pyplot as plt
import numpy as np
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.overlay import NdOverlay
from holoviews.core.spaces import HoloMap
from holoviews.element import Points
from ..utils import ParamLogStream
from .test_plot import TestMPLPlot, mpl_renderer
def test_points_rcparams_used(self):
    opts = dict(fig_rcparams={'grid.color': 'red'})
    points = Points(([0, 1], [0, 3])).opts(**opts)
    plot = mpl_renderer.get_plot(points)
    ax = plot.state.axes[0]
    lines = ax.get_xgridlines()
    self.assertEqual(lines[0].get_color(), 'red')