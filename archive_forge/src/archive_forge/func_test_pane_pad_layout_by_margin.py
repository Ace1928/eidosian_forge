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
def test_pane_pad_layout_by_margin():
    md = Markdown(width=300, height=300, margin=(25, 12, 14, 42))
    assert md.layout.width == 354
    assert md.layout.height == 339