import param
import pytest
import panel as pn
from panel.chat import ChatMessage
from panel.config import config
from panel.interact import interactive
from panel.io.loading import LOADING_INDICATOR_CSS_CLASS
from panel.layout import Row
from panel.links import CallbackGenerator
from panel.pane import (
from panel.param import (
from panel.tests.util import check_layoutable_properties
from panel.util import param_watchers
@pytest.mark.parametrize('pane', all_panes + [Bokeh])
def test_pane_loading_param(pane, document, comm):
    try:
        p = pane()
    except ImportError:
        pytest.skip('Dependent library could not be imported.')
    root = p.get_root(document, comm)
    model = p._models[root.ref['id']][0]
    p.loading = True
    css_classes = [LOADING_INDICATOR_CSS_CLASS, f'pn-{config.loading_spinner}']
    assert all((cls in model.css_classes for cls in css_classes))
    p.loading = False
    assert not any((cls in model.css_classes for cls in css_classes))