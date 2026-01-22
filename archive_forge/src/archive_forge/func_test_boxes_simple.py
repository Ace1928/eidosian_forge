import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
def test_boxes_simple(self):
    boxes = Rectangles([(0, 0, 1, 1), (2, 2, 4, 3)])
    state = self._get_plot_state(boxes)
    shapes = state['layout']['shapes']
    self.assertEqual(len(shapes), 2)
    self.assertEqual(shapes[0], {'type': 'rect', 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1, 'xref': 'x', 'yref': 'y', 'name': '', 'line': {'color': default_shape_color}})
    self.assertEqual(shapes[1], {'type': 'rect', 'x0': 2, 'y0': 2, 'x1': 4, 'y1': 3, 'xref': 'x', 'yref': 'y', 'name': '', 'line': {'color': default_shape_color}})
    self.assertEqual(state['layout']['xaxis']['range'], [0, 4])
    self.assertEqual(state['layout']['yaxis']['range'], [0, 3])