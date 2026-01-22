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
def test_element_yaxis_right_bare(self):
    curve = Curve(range(10)).opts(yaxis='right-bare')
    plot = bokeh_renderer.get_plot(curve)
    yaxis = plot.handles['yaxis']
    self.assertEqual(yaxis.axis_label_text_font_size, '0pt')
    self.assertEqual(yaxis.major_label_text_font_size, '0pt')
    self.assertEqual(yaxis.minor_tick_line_color, None)
    self.assertEqual(yaxis.major_tick_line_color, None)
    self.assertTrue(yaxis in plot.state.right)