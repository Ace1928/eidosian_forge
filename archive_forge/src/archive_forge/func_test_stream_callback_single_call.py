from collections import deque
import numpy as np
import pandas as pd
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Path3D, Scatter3D
from holoviews.streams import PointerX
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_stream_callback_single_call(self):
    history = deque(maxlen=10)

    def history_callback(x):
        history.append(x)
        return Curve(list(history))
    stream = PointerX(x=0)
    dmap = DynamicMap(history_callback, kdims=[], streams=[stream])
    plot = plotly_renderer.get_plot(dmap)
    plotly_renderer(dmap)
    for i in range(20):
        stream.event(x=i)
    state = plot.state
    self.assertEqual(state['data'][0]['x'], np.arange(10))
    self.assertEqual(state['data'][0]['y'], np.arange(10, 20))