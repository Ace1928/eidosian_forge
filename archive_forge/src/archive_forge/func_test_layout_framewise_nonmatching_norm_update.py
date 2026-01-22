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
def test_layout_framewise_nonmatching_norm_update(self):
    img1 = Image(np.mgrid[0:5, 0:5][0], vdims='z').opts(framewise=True)
    stream = Stream.define('zscale', value=1)()
    transform = dim('z2') * stream.param.value
    img2 = Image(np.mgrid[0:5, 0:5][0], vdims='z2').apply.transform(z2=transform).opts(framewise=True)
    plot = bokeh_renderer.get_plot(img1 + img2)
    img1_plot = plot.subplots[0, 0].subplots['main']
    img2_plot = plot.subplots[0, 1].subplots['main']
    img1_cmapper = img1_plot.handles['color_mapper']
    img2_cmapper = img2_plot.handles['color_mapper']
    self.assertEqual(img1_cmapper.low, 0)
    self.assertEqual(img2_cmapper.low, 0)
    self.assertEqual(img1_cmapper.high, 4)
    self.assertEqual(img2_cmapper.high, 4)
    stream.update(value=10)
    self.assertEqual(img1_cmapper.high, 4)
    self.assertEqual(img2_cmapper.high, 40)
    stream.update(value=2)
    self.assertEqual(img1_cmapper.high, 4)
    self.assertEqual(img2_cmapper.high, 8)