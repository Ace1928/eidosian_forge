import pytest
from packaging.version import Version
import numpy as np
import panel as pn
from panel.models.vega import VegaPlot
from panel.pane import PaneBase, Vega
def test_vega_pane_inline(document, comm):
    pane = pn.panel(vega_inline_example)
    model = pane.get_root(document, comm=comm)
    assert isinstance(model, VegaPlot)
    assert dict(model.data, **blank_schema) == dict(vega_inline_example, **blank_schema)
    assert model.data_sources == {}
    pane._cleanup(model)
    assert pane._models == {}