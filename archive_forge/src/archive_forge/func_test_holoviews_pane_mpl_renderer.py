import datetime as dt
import warnings
import numpy as np
import pytest
from bokeh.models import (
from bokeh.plotting import figure
import panel as pn
from panel.depends import bind
from panel.layout import (
from panel.pane import HoloViews, PaneBase, panel
from panel.tests.util import hv_available, mpl_available
from panel.theme import Native
from panel.util.warnings import PanelDeprecationWarning
from panel.widgets import (
@pytest.mark.usefixtures('hv_mpl')
@mpl_available
@hv_available
def test_holoviews_pane_mpl_renderer(document, comm):
    curve = hv.Curve([1, 2, 3])
    pane = pn.panel(curve)
    row = pane.get_root(document, comm=comm)
    assert isinstance(row, BkRow)
    assert len(row.children) == 1
    model = row.children[0]
    assert pane._models[row.ref['id']][0] is model
    assert model.text.startswith('&lt;img src=')
    scatter = hv.Scatter([1, 2, 3])
    pane.object = scatter
    new_model = row.children[0]
    assert model.text != new_model.text
    pane._cleanup(row)
    assert pane._models == {}