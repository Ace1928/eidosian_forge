import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
def test_bounds_layout(self):
    box1 = Box(0, 0, (1, 1), orientation=0)
    box2 = Box(0, 0, (2, 2), orientation=0.5)
    box3 = Box(0, 0, (3, 3), orientation=1.0)
    box4 = Box(0, 0, (4, 4), orientation=1.5)
    layout = (box1 + box2 + box3 + box4).cols(2)
    state = self._get_plot_state(layout)
    shapes = state['layout']['shapes']
    self.assertEqual(len(shapes), 4)
    self.assert_path_shape_element(shapes[0], box3, xref='x', yref='y')
    self.assert_path_shape_element(shapes[1], box4, xref='x2', yref='y2')
    self.assert_path_shape_element(shapes[2], box1, xref='x3', yref='y3')
    self.assert_path_shape_element(shapes[3], box2, xref='x4', yref='y4')