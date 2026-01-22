import datetime as dt
import numpy as np
import pandas as pd
from holoviews.element import Area, Overlay
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_area_with_nans(self):
    area = Area([1, 2, 3, np.nan, 5, 6, 7])
    plot = bokeh_renderer.get_plot(area)
    cds = plot.handles['cds']
    self.assertEqual(cds.data['x'], np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0, np.nan, 4.0, 5.0, 6.0, 6.0, 5.0, 4.0]))
    self.assertEqual(cds.data['y'], np.array([0.0, 0.0, 0.0, 3.0, 2.0, 1.0, np.nan, 0.0, 0.0, 0.0, 7.0, 6.0, 5.0]))