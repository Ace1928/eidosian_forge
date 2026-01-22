from itertools import product
import numpy as np
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from .test_plot import TestMPLPlot, mpl_renderer
def test_get_data_yseparators(self):
    plot = mpl_renderer.get_plot(self.element.opts(ymarks=4))
    data, style, ticks = plot.get_data(self.element, {'z': {'combined': (0, 3)}}, {})
    yseparators = data['yseparator']
    for circle, r in zip(yseparators, [0.25, 0.375]):
        self.assertEqual(circle.radius, r)