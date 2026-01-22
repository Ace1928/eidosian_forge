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
@pytest.mark.parametrize('pane', all_panes)
def test_pane_clone(pane):
    try:
        p = pane()
    except ImportError:
        pytest.skip('Dependent library could not be imported.')
    clone = p.clone()
    assert [(k, v) for k, v in sorted(p.param.values().items()) if k not in ('name', '_pane')] == [(k, v) for k, v in sorted(clone.param.values().items()) if k not in ('name', '_pane')]