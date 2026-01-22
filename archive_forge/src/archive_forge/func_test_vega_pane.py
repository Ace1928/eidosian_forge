import pytest
from packaging.version import Version
import numpy as np
import panel as pn
from panel.models.vega import VegaPlot
from panel.pane import PaneBase, Vega
def test_vega_pane(document, comm):
    pane = pn.panel(vega_example)
    model = pane.get_root(document, comm=comm)
    assert isinstance(model, VegaPlot)
    expected = dict(vega_example, data={})
    assert dict(model.data, **blank_schema) == dict(expected, **blank_schema)
    cds_data = model.data_sources['data'].data
    assert np.array_equal(cds_data['x'], np.array(['A', 'B', 'C', 'D', 'E']))
    assert np.array_equal(cds_data['y'], np.array([5, 3, 6, 7, 2]))
    point_example = dict(vega_example, mark='point')
    point_example['data']['values'][0]['x'] = 'C'
    pane.object = point_example
    point_example = dict(point_example, data={})
    assert model.data == point_example
    cds_data = model.data_sources['data'].data
    assert np.array_equal(cds_data['x'], np.array(['C', 'B', 'C', 'D', 'E']))
    assert np.array_equal(cds_data['y'], np.array([5, 3, 6, 7, 2]))
    pane._cleanup(model)
    assert pane._models == {}