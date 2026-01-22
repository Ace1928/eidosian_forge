import numpy as np
from bokeh.models import FactorRange, HoverTool, Range1d
from holoviews.element import HeatMap, Image, Points
from .test_plot import TestBokehPlot, bokeh_renderer
def test_heatmap_points_categorical_axes_string_int(self):
    hmap = HeatMap([('A', 1, 1), ('B', 2, 2)])
    points = Points([('A', 2), ('B', 1), ('C', 3)])
    plot = bokeh_renderer.get_plot(hmap * points)
    x_range = plot.handles['x_range']
    y_range = plot.handles['y_range']
    self.assertIsInstance(x_range, FactorRange)
    self.assertEqual(x_range.factors, ['A', 'B', 'C'])
    self.assertIsInstance(y_range, Range1d)
    self.assertEqual(y_range.start, 0.5)
    self.assertEqual(y_range.end, 3)