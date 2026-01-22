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
def test_layout_title_fontsize(self):
    hmap1 = HoloMap({a: Image(np.random.rand(10, 10)) for a in range(3)})
    hmap2 = HoloMap({a: Image(np.random.rand(10, 10)) for a in range(3)})
    layout = Layout([hmap1, hmap2]).opts(fontsize={'title': '12pt'})
    plot = bokeh_renderer.get_plot(layout)
    title = plot.handles['title']
    self.assertIsInstance(title, Div)
    text = '<span style="color:black;font-family:Arial;font-style:bold;font-weight:bold;font-size:12pt">Default: 0</span>'
    self.assertEqual(title.text, text)