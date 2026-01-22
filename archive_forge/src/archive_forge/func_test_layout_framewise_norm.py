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
def test_layout_framewise_norm(self):
    img1 = Image(np.mgrid[0:5, 0:5][0]).opts(framewise=True)
    img2 = Image(np.mgrid[0:5, 0:5][0] * 10).opts(framewise=True)
    plot = bokeh_renderer.get_plot(img1 + img2)
    img1_plot, img2_plot = (sp.subplots['main'] for sp in plot.subplots.values())
    img1_cmapper = img1_plot.handles['color_mapper']
    img2_cmapper = img2_plot.handles['color_mapper']
    self.assertEqual(img1_cmapper.low, 0)
    self.assertEqual(img2_cmapper.low, 0)
    self.assertEqual(img1_cmapper.high, 40)
    self.assertEqual(img2_cmapper.high, 40)