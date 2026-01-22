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
def test_categorical_overlay_dimension_values(self):
    curve = Curve([('C', 1), ('B', 3)]).redim.values(x=['A', 'B', 'C'])
    scatter = Scatter([('A', 2)])
    plot = bokeh_renderer.get_plot(curve * scatter)
    x_range = plot.handles['x_range']
    self.assertEqual(x_range.factors, ['A', 'B', 'C'])