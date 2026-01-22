import datetime as dt
import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, FactorRange, LinearColorMapper, Scatter
from holoviews.core import NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Points
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import Stream
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_points_overlay_categorical_xaxis_invert_axes(self):
    points = Points((['A', 'B', 'C'], (1, 2, 3))).opts(invert_axes=True)
    points2 = Points((['B', 'C', 'D'], (1, 2, 3)))
    plot = bokeh_renderer.get_plot(points * points2)
    y_range = plot.handles['y_range']
    self.assertIsInstance(y_range, FactorRange)
    self.assertEqual(y_range.factors, ['A', 'B', 'C', 'D'])