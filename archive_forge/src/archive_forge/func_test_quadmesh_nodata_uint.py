import numpy as np
from holoviews.element import QuadMesh
from .test_plot import TestPlotlyPlot
def test_quadmesh_nodata_uint(self):
    img = QuadMesh(([1, 2, 4], [0, 1], np.array([[0, 1, 2], [2, 3, 4]], dtype='uint32'))).opts(nodata=0)
    state = self._get_plot_state(img)
    self.assertEqual(state['data'][0]['type'], 'heatmap')
    self.assertEqual(state['data'][0]['z'], np.array([[np.nan, 1, 2], [2, 3, 4]]))