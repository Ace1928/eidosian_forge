import datetime as dt
import numpy as np
import pandas as pd
from holoviews.element import Area, Overlay
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_area_with_lower_vdim(self):
    area = Area([(1, 0.5, 1), (2, 1.5, 2), (3, 2.5, 3)], vdims=['y', 'y2']).opts(padding=0.1)
    plot = bokeh_renderer.get_plot(area)
    x_range, y_range = (plot.handles['x_range'], plot.handles['y_range'])
    self.assertEqual(x_range.start, 0.8)
    self.assertEqual(x_range.end, 3.2)
    self.assertEqual(y_range.start, 0.25)
    self.assertEqual(y_range.end, 3.25)