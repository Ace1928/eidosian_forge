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
def test_element_xaxis_bare(self):
    curve = Curve(range(10)).opts(xaxis='bare')
    plot = bokeh_renderer.get_plot(curve)
    xaxis = plot.handles['xaxis']
    self.assertEqual(xaxis.axis_label_text_font_size, '0pt')
    self.assertEqual(xaxis.major_label_text_font_size, '0pt')
    self.assertEqual(xaxis.minor_tick_line_color, None)
    self.assertEqual(xaxis.major_tick_line_color, None)
    self.assertTrue(xaxis in plot.state.below)