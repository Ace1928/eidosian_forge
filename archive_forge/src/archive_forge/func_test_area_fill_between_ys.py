import numpy as np
import pandas as pd
from holoviews.element import Area, Overlay
from .test_plot import TestPlotlyPlot
def test_area_fill_between_ys(self):
    area = Area([(0, 1, 0.5), (1, 2, 1), (2, 3, 2.25)], vdims=['y', 'y2'])
    state = self._get_plot_state(area)
    self.assertEqual(state['data'][0]['y'], np.array([0.5, 1, 2.25]))
    self.assertEqual(state['data'][0]['mode'], 'lines')
    self.assertEqual(state['data'][0].get('fill', None), None)
    self.assertEqual(state['data'][1]['y'], np.array([1, 2, 3]))
    self.assertEqual(state['data'][1]['mode'], 'lines')
    self.assertEqual(state['data'][1]['fill'], 'tonexty')
    self.assertEqual(state['layout']['yaxis']['range'], [0.5, 3])