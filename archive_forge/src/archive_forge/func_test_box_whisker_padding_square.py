import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, ColumnDataSource, LinearColorMapper
from holoviews.element import BoxWhisker
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_box_whisker_padding_square(self):
    curve = BoxWhisker([1, 2, 3]).opts(padding=0.1)
    plot = bokeh_renderer.get_plot(curve)
    y_range = plot.handles['y_range']
    self.assertEqual(y_range.start, 0.8)
    self.assertEqual(y_range.end, 3.2)