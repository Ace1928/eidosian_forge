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
def test_element_font_scaling(self):
    curve = Curve(range(10)).opts(fontscale=2, title='A title')
    plot = bokeh_renderer.get_plot(curve)
    fig = plot.state
    xaxis = plot.handles['xaxis']
    yaxis = plot.handles['yaxis']
    self.assertEqual(fig.title.text_font_size, '24pt')
    self.assertEqual(xaxis.axis_label_text_font_size, '26px')
    self.assertEqual(yaxis.axis_label_text_font_size, '26px')
    self.assertEqual(xaxis.major_label_text_font_size, '22px')
    self.assertEqual(yaxis.major_label_text_font_size, '22px')