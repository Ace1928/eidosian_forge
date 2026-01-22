import pytest
from packaging.version import Version
import numpy as np
import panel as pn
from panel.models.vega import VegaPlot
from panel.pane import PaneBase, Vega
def test_vega_lite_4_selection_spec(document, comm):
    vega = Vega(vega4_selection_example)
    assert vega._selections == {'brush': 'interval'}