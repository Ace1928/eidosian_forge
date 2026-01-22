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
def test_get_root(document, comm):

    class Test(param.Parameterized):
        pass
    test = Test()
    test_pane = Param(test)
    model = test_pane.get_root(document, comm=comm)
    assert isinstance(model, BkColumn)
    assert len(model.children) == 1
    html = model.children[0]
    assert isinstance(html, Div)
    assert html.text == '<b>' + test.name[:-5] + '</b>'