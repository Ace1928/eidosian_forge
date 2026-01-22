import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
def test_vline_layout(self):
    layout = (VLine(1) + VLine(2) + VLine(3) + VLine(4)).cols(2).opts(vspacing=0, hspacing=0)
    state = self._get_plot_state(layout)
    shapes = state['layout']['shapes']
    self.assertEqual(len(shapes), 4)
    self.assert_vline(shapes[0], 3, xref='x', ydomain=[0.0, 0.5])
    self.assert_vline(shapes[1], 4, xref='x2', ydomain=[0.0, 0.5])
    self.assert_vline(shapes[2], 1, xref='x3', ydomain=[0.5, 1.0])
    self.assert_vline(shapes[3], 2, xref='x4', ydomain=[0.5, 1.0])