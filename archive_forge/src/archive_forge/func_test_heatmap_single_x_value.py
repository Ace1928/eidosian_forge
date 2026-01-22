import numpy as np
from bokeh.models import FactorRange, HoverTool, Range1d
from holoviews.element import HeatMap, Image, Points
from .test_plot import TestBokehPlot, bokeh_renderer
def test_heatmap_single_x_value(self):
    hmap = HeatMap(([1], ['A', 'B'], np.array([[1], [2]])))
    plot = bokeh_renderer.get_plot(hmap)
    cds = plot.handles['cds']
    self.assertEqual(cds.data['x'], np.array([1, 1]))
    self.assertEqual(cds.data['y'], np.array(['A', 'B']))
    self.assertEqual(cds.data['width'], [2.0, 2.0])
    self.assertEqual(plot.handles['glyph'].height, 1)