import numpy as np
from holoviews.element import Curve, Tiles
from .test_plot import TestPlotlyPlot
def test_curve_dash(self):
    curve = Tiles('') * Curve([1, 2, 3]).opts(dash='dash')
    with self.assertRaises(ValueError) as e:
        self._get_plot_state(curve)
    self.assertIn('dash', str(e.exception))