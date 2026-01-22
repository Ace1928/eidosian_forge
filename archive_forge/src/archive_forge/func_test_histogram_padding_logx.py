import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, DatetimeAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay
from holoviews.element import Dataset, Histogram, Image, Points
from holoviews.operation import histogram
from holoviews.plotting.bokeh.util import property_to_dict
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_histogram_padding_logx(self):
    histogram = Histogram([(1, 1), (2, 2), (3, 3)]).opts(padding=0.1, logx=True)
    plot = bokeh_renderer.get_plot(histogram)
    x_range, y_range = (plot.handles['x_range'], plot.handles['y_range'])
    self.assertEqual(x_range.start, 0.41158562699652224)
    self.assertEqual(x_range.end, 4.251849154136733)
    self.assertEqual(y_range.start, 0)
    self.assertEqual(y_range.end, 3.2)