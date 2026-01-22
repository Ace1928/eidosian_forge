import numpy as np
import pytest
from bokeh.core.serialization import Serializer
from panel.models.deckgl import DeckGLPlot
from panel.pane import DeckGL, PaneBase, panel
def test_deckgl_construct_layer(document, comm):
    pane = DeckGL({'layers': [{'data': [{'a': 1, 'b': 2}, {'a': 3, 'b': 7}]}]})
    model = pane.get_root(document, comm)
    assert model.layers == [{'data': 0}]
    assert len(model.data_sources) == 1
    data = model.data_sources[0].data
    assert np.array_equal(data['a'], np.array([1, 3]))
    assert np.array_equal(data['b'], np.array([2, 7]))