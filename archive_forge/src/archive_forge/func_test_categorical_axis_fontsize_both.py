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
def test_categorical_axis_fontsize_both(self):
    curve = Curve([('A', 1), ('B', 2)]).opts(fontsize={'xticks': 18})
    plot = bokeh_renderer.get_plot(curve)
    xaxis = plot.handles['xaxis']
    self.assertEqual(xaxis.major_label_text_font_size, '18pt')
    self.assertEqual(xaxis.group_text_font_size, '18pt')