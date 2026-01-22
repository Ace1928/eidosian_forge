import datetime as dt
import pytest
import numpy as np
import pandas as pd
from panel.models.plotly import PlotlyPlot
from panel.pane import PaneBase, Plotly
@plotly_available
def test_plotly_pane_datetime64_array_transform(document, comm):
    index = np.array([dt.datetime(2019, 1, i) for i in range(1, 11)]).astype('M8[us]')
    data = np.random.randn(10)
    traces = [go.Scatter(x=index, y=data)]
    fig = go.Figure(traces)
    pane = Plotly(fig)
    model = pane.get_root(document, comm)
    assert model.data_sources[0].data['x'][0].dtype.kind in 'SU'