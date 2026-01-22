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
def test_replace_param_object(document, comm):

    class Test(param.Parameterized):
        a = param.Number(bounds=(0, 10))
    pane = Param()
    model = pane.get_root(document, comm=comm)
    assert model.children == []
    pane.object = Test()
    assert len(model.children) == 2
    title, widget = model.children
    assert isinstance(title, Div)
    assert title.text == '<b>Test</b>'
    assert isinstance(widget, Slider)
    assert widget.start == 0
    assert widget.end == 10
    pane.object = Test().param
    assert len(model.children) == 2
    title, widget = model.children
    assert isinstance(title, Div)
    assert title.text == '<b>Test</b>'
    assert isinstance(widget, Slider)
    assert widget.start == 0
    assert widget.end == 10
    pane.object = None
    assert len(model.children) == 0