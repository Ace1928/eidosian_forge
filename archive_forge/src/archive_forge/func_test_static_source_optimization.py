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
def test_static_source_optimization(self):
    data = np.ones((5, 5))
    img = Image(data)

    def get_img(test):
        get_img.data *= test
        return img
    get_img.data = data
    stream = Stream.define('Test', test=1)()
    dmap = DynamicMap(get_img, streams=[stream])
    plot = bokeh_renderer.get_plot(dmap, doc=Document())
    source = plot.handles['source']
    self.assertEqual(source.data['image'][0].mean(), 1)
    stream.event(test=2)
    self.assertTrue(plot.static_source)
    self.assertEqual(source.data['image'][0].mean(), 2)
    self.assertNotIn(source, plot.current_handles)