from collections import deque
import numpy as np
import pandas as pd
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Path3D, Scatter3D
from holoviews.streams import PointerX
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_element_plot3d_padding(self):
    scatter = Scatter3D([(0, 1, 2), (1, 2, 3), (2, 3, 5)]).opts(padding=0.1)
    state = self._get_plot_state(scatter)
    self.assertEqual(state['layout']['scene']['xaxis']['range'], [-0.2, 2.2])
    self.assertEqual(state['layout']['scene']['yaxis']['range'], [0.8, 3.2])
    self.assertEqual(state['layout']['scene']['zaxis']['range'], [1.7, 5.3])