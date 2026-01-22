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
def test_set_name(document, comm):

    class Test(param.Parameterized):
        a = param.Number(bounds=(0, 10))
        b = param.String(default='A')
    pane = Param(Test(), name='First')
    model = pane.get_root(document, comm=comm)
    assert len(model.children) == 3
    title, slider, text = model.children
    assert isinstance(title, Div)
    assert title.text == '<b>First</b>'
    assert isinstance(slider, Slider)
    assert isinstance(text, TextInput)
    pane.name = 'Second'
    assert len(model.children) == 3
    title, _, _ = model.children
    assert isinstance(title, Div)
    assert title.text == '<b>Second</b>'