import pytest
from packaging.version import Version
import numpy as np
import panel as pn
from panel.models.vega import VegaPlot
from panel.pane import PaneBase, Vega
def test_get_vega_pane_type_from_dict():
    assert PaneBase.get_pane_type(vega_example) is Vega