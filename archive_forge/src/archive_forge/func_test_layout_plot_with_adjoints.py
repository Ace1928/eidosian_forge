import datetime as dt
import re
import numpy as np
from bokeh.models import Div, GlyphRenderer, GridPlot, Spacer, Tabs, Title, Toolbar
from bokeh.models.layouts import TabPanel
from bokeh.plotting import figure
from holoviews.core import (
from holoviews.element import Curve, Histogram, Image, Points, Scatter
from holoviews.streams import Stream
from holoviews.util import opts, render
from holoviews.util.transform import dim
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_layout_plot_with_adjoints(self):
    layout = (Curve([]) + Curve([]).hist()).cols(1)
    plot = bokeh_renderer.get_plot(layout)
    grid = plot.state
    toolbar = grid.toolbar
    self.assertIsInstance(toolbar, Toolbar)
    self.assertIsInstance(grid, GridPlot)
    for fig, _, _ in grid.children:
        self.assertIsInstance(fig, figure)
    self.assertTrue([len([r for r in f.renderers if isinstance(r, GlyphRenderer)]) for f, _, _ in grid.children], [1, 1, 1])