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
def test_skip_param(document, comm):
    checkbox = Checkbox(value=False)
    button = Button()

    def layout(value, click):
        if not click:
            raise Skip()
        return Markdown(f'{value}')
    layout = ParamFunction(bind(layout, checkbox, button))
    root = layout.get_root(document, comm)
    div = root.children[0]
    assert div.text == '&lt;pre&gt; &lt;/pre&gt;'
    checkbox.value = True
    assert div.text == '&lt;pre&gt; &lt;/pre&gt;'
    button.param.trigger('value')
    assert div.text == '&lt;pre&gt; &lt;/pre&gt;'