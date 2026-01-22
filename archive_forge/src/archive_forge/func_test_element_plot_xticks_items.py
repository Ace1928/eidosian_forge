from collections import deque
import numpy as np
import pandas as pd
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Path3D, Scatter3D
from holoviews.streams import PointerX
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_element_plot_xticks_items(self):
    curve = Curve([(1, 1), (5, 2), (10, 3)]).opts(xticks=[(1, 'A'), (5, 'B'), (10, 'C')])
    state = self._get_plot_state(curve)
    self.assertEqual(state['layout']['xaxis']['tickvals'], [1, 5, 10])
    self.assertEqual(state['layout']['xaxis']['ticktext'], ['A', 'B', 'C'])