import numpy as np
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels, Tiles
from .test_plot import TestPlotlyPlot
def test_labels_inverted(self):
    labels = Tiles('') * Labels([(0, 3, 0), (1, 2, 1), (2, 1, 1)]).opts(invert_axes=True)
    with self.assertRaises(ValueError) as e:
        self._get_plot_state(labels)
    self.assertIn('invert_axes', str(e.exception))