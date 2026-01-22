import numpy as np
from holoviews.element import Scatter, Tiles
from .test_plot import TestPlotlyPlot
def test_scatter_size(self):
    scatter = Tiles('') * Scatter([3, 2, 1]).opts(size='y')
    state = self._get_plot_state(scatter)
    self.assertEqual(state['data'][1]['marker']['size'], np.array([3, 2, 1]))