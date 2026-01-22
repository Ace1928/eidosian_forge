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
def test_empty_adjoint_plot(self):
    adjoint = Curve([0, 1, 1, 2, 3]) << Empty() << Curve([0, 1, 1, 0, 1])
    plot = bokeh_renderer.get_plot(adjoint)
    adjoint_plot = plot.subplots[0, 0]
    self.assertEqual(len(adjoint_plot.subplots), 3)
    grid = plot.state
    (f1, *_), (f2, *_), (s1, *_) = grid.children
    self.assertIsInstance(grid, GridPlot)
    self.assertIsInstance(s1, Spacer)
    self.assertEqual(s1.width, 0)
    self.assertEqual(s1.height, 0)
    self.assertEqual(f1.height, f2.height)
    self.assertEqual(f1.height, 300)