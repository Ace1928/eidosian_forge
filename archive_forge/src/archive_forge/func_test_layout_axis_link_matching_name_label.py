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
def test_layout_axis_link_matching_name_label(self):
    layout = Curve([1, 2, 3], vdims=('a', 'A')) + Curve([1, 2, 3], vdims=('a', 'A'))
    plot = bokeh_renderer.get_plot(layout)
    p1, p2 = (sp.subplots['main'] for sp in plot.subplots.values())
    self.assertIs(p1.handles['y_range'], p2.handles['y_range'])