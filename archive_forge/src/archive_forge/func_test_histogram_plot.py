import numpy as np
from holoviews.element import Histogram
from .test_plot import TestPlotlyPlot
def test_histogram_plot(self):
    hist = Histogram((self.edges, self.frequencies))
    state = self._get_plot_state(hist)
    np.testing.assert_equal(state['data'][0]['x'], self.edges)
    np.testing.assert_equal(state['data'][0]['y'], self.frequencies)
    self.assertEqual(state['data'][0]['type'], 'bar')
    self.assertEqual(state['data'][0]['orientation'], 'v')
    self.assertEqual(state['data'][0]['width'], 1)
    self.assertEqual(state['layout']['xaxis']['range'], [-3.5, 2.5])
    self.assertEqual(state['layout']['xaxis']['title']['text'], 'x')
    self.assertEqual(state['layout']['yaxis']['range'], [0, 5])
    self.assertEqual(state['layout']['yaxis']['title']['text'], 'Frequency')