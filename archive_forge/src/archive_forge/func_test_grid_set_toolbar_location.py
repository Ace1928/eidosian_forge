import numpy as np
from bokeh.layouts import Column
from bokeh.models import Div, Toolbar
from holoviews.core import (
from holoviews.element import Curve, Image, Points
from holoviews.operation import gridmatrix
from holoviews.streams import Stream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_grid_set_toolbar_location(self):
    grid = GridSpace({0: Curve([]), 1: Points([])}, 'X').opts(toolbar='left')
    plot = bokeh_renderer.get_plot(grid)
    self.assertIsInstance(plot.state, Column)
    self.assertIsInstance(plot.state.children[0].toolbar, Toolbar)