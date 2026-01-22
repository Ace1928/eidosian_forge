from itertools import product
import numpy as np
from bokeh.models import ColorBar
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from holoviews.plotting.bokeh import RadialHeatMapPlot
from .test_plot import TestBokehPlot, bokeh_renderer
def test_get_seg_labels_data(self):
    """Test correct computation of a single segment label data point.

        """
    radiant = np.pi / 2
    x = np.cos(radiant) + 1
    y = np.sin(radiant) + 1
    angle = 1.5 * np.pi + radiant
    test_seg_data = dict(x=np.array(x), y=np.array(y), text=np.array('o1'), angle=np.array(angle))
    cmp_seg_data = self.plot._get_seg_labels_data(['o1'], self.seg_bins)
    self.assertEqual(test_seg_data, cmp_seg_data)