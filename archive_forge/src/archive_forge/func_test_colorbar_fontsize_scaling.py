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
def test_colorbar_fontsize_scaling(self):
    img = Image(np.array([[0, 1], [2, 3]])).opts(colorbar=True, fontscale=2)
    plot = bokeh_renderer.get_plot(img)
    colorbar = plot.handles['colorbar']
    self.assertEqual(colorbar.title_text_font_size, '26px')
    self.assertEqual(colorbar.major_label_text_font_size, '22px')