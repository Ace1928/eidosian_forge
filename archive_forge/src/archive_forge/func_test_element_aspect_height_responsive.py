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
def test_element_aspect_height_responsive(self):
    curve = Curve([1, 2, 3]).opts(aspect=2, height=400, responsive=True)
    plot = bokeh_renderer.get_plot(curve)
    self.assertEqual(plot.state.frame_height, 400)
    self.assertEqual(plot.state.frame_width, 800)
    self.log_handler.assertContains('WARNING', 'responsive mode could not be enabled')
    self.assertEqual(plot.state.height, None)
    self.assertEqual(plot.state.width, None)
    self.assertEqual(plot.state.sizing_mode, 'fixed')
    self.log_handler.assertContains('WARNING', 'uses those values as frame_width/frame_height instead')