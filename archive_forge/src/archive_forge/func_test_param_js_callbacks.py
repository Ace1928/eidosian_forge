import asyncio
import os
import pandas as pd
import param
import pytest
from bokeh.models import (
from packaging.version import Version
from panel import config
from panel.depends import bind
from panel.io.state import set_curdoc, state
from panel.layout import Row, Tabs
from panel.models import HTML as BkHTML
from panel.pane import (
from panel.param import (
from panel.tests.util import mpl_available, mpl_figure
from panel.widgets import (
def test_param_js_callbacks(document, comm):

    class JsButton(param.Parameterized):
        param_btn = param.Action(default=lambda self: print('Action Python Response'), label='Action')
    param_button = Param(JsButton())
    code = "console.log('Action button clicked')"
    param_button[1].js_on_click(code=code)
    model = param_button.get_root(document, comm=comm)
    button = model.children[1]
    assert len(button.js_event_callbacks) == 1
    callbacks = button.js_event_callbacks
    assert 'button_click' in callbacks
    assert len(callbacks['button_click']) == 1
    assert code in callbacks['button_click'][0].code