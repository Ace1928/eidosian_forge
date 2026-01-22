import numpy as np
import pytest
from bokeh.core.serialization import Serializer
from panel.models.deckgl import DeckGLPlot
from panel.pane import DeckGL, PaneBase, panel
@pydeck_available
def test_pydeck_mapbox_api_key_issue_5790(document, comm):
    deck_wo_key = pydeck.Deck()
    pane_w_key = DeckGL(deck_wo_key, mapbox_api_key='ABC')
    model = pane_w_key.get_root(document, comm=comm)
    assert model.mapbox_api_key == 'ABC'