import datetime as dt
from unittest import SkipTest
import numpy as np
import panel as pn
import pytest
from bokeh.document import Document
from bokeh.models import (
from holoviews.core import DynamicMap, HoloMap, NdOverlay
from holoviews.core.util import dt_to_int
from holoviews.element import Curve, HeatMap, Image, Labels, Scatter
from holoviews.plotting.util import process_cmap
from holoviews.streams import PointDraw, Stream
from holoviews.util import render
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_element_grid_options(self):
    grid_style = {'grid_line_color': 'blue', 'grid_line_width': 1.5, 'ygrid_bounds': (0.3, 0.7), 'minor_xgrid_line_color': 'lightgray', 'xgrid_line_dash': [4, 4]}
    curve = Curve(range(10)).opts(show_grid=True, gridstyle=grid_style)
    plot = bokeh_renderer.get_plot(curve)
    self.assertEqual(plot.state.xgrid[0].grid_line_color, 'blue')
    self.assertEqual(plot.state.xgrid[0].grid_line_width, 1.5)
    self.assertEqual(plot.state.xgrid[0].grid_line_dash, [4, 4])
    self.assertEqual(plot.state.xgrid[0].minor_grid_line_color, 'lightgray')
    self.assertEqual(plot.state.ygrid[0].grid_line_color, 'blue')
    self.assertEqual(plot.state.ygrid[0].grid_line_width, 1.5)
    self.assertEqual(plot.state.ygrid[0].bounds, (0.3, 0.7))