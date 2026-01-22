from collections import deque
import numpy as np
import pandas as pd
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Path3D, Scatter3D
from holoviews.streams import PointerX
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_element_plot_xlim(self):
    curve = Curve([(1, 1), (2, 10), (3, 100)]).opts(xlim=(0, 1010))
    state = self._get_plot_state(curve)
    self.assertEqual(state['layout']['xaxis']['range'], [0, 1010])