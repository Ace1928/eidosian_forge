import numpy as np
import pytest
from bokeh.core.serialization import Serializer
from panel.models.deckgl import DeckGLPlot
from panel.pane import DeckGL, PaneBase, panel
@pydeck_available
def test_pydeck_no_min_max_zoom_issue_5790(document, comm):
    state_w_no_min_max_zoom = {'latitude': 37.7749, 'longitude': -122.4194, 'zoom': 10, 'bearing': 0, 'pitch': 0}
    view_state = pydeck.ViewState(**state_w_no_min_max_zoom)
    deck = pydeck.Deck(initial_view_state=view_state)
    pane = DeckGL(deck)
    model = pane.get_root(document, comm=comm)
    assert model.initialViewState == state_w_no_min_max_zoom