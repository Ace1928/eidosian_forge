import datetime as dt
import pytest
import numpy as np
import pandas as pd
from panel.models.plotly import PlotlyPlot
from panel.pane import PaneBase, Plotly
@plotly_available
def test_plotly_pane_numpy_to_cds_traces(document, comm):
    trace = go.Scatter(x=np.array([1, 2]), y=np.array([2, 3]))
    pane = Plotly({'data': [trace], 'layout': {'width': 350}})
    model = pane.get_root(document, comm=comm)
    assert isinstance(model, PlotlyPlot)
    assert len(model.data) == 1
    assert model.data[0]['type'] == 'scatter'
    assert 'x' not in model.data[0]
    assert 'y' not in model.data[0]
    assert model.layout == {'width': 350}
    assert len(model.data_sources) == 1
    cds = model.data_sources[0]
    assert np.array_equal(cds.data['x'][0], np.array([1, 2]))
    assert np.array_equal(cds.data['y'][0], np.array([2, 3]))
    new_trace = [go.Scatter(x=np.array([5, 6]), y=np.array([6, 7])), go.Bar(x=np.array([2, 3]), y=np.array([4, 5]))]
    pane.object = {'data': new_trace, 'layout': {'width': 350}}
    assert len(model.data) == 2
    assert model.data[0]['type'] == 'scatter'
    assert 'x' not in model.data[0]
    assert 'y' not in model.data[0]
    assert model.data[1]['type'] == 'bar'
    assert 'x' not in model.data[1]
    assert 'y' not in model.data[1]
    assert model.layout == {'width': 350}
    assert len(model.data_sources) == 2
    cds = model.data_sources[0]
    assert np.array_equal(cds.data['x'][0], np.array([5, 6]))
    assert np.array_equal(cds.data['y'][0], np.array([6, 7]))
    cds2 = model.data_sources[1]
    assert np.array_equal(cds2.data['x'][0], np.array([2, 3]))
    assert np.array_equal(cds2.data['y'][0], np.array([4, 5]))
    pane._cleanup(model)
    assert pane._models == {}