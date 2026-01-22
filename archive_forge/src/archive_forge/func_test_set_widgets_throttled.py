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
def test_set_widgets_throttled(document, comm):

    class Test(param.Parameterized):
        a = param.Number(default=0, bounds=(0, 10), precedence=1)
    test = Test()
    pane = Param(test)
    model = pane.get_root(document, comm=comm)
    pane.widgets = {'a': {'throttled': False}}
    assert len(model.children) == 2
    _, number = model.children
    number.value = 1
    assert number.value == 1
    assert number.value_throttled != 1
    assert test.a == 1
    test.a = 2
    assert number.value == 2
    assert number.value_throttled != 2
    assert test.a == 2
    pane.widgets = {'a': {'throttled': True}}
    assert len(model.children) == 2
    _, number = model.children
    pane._widgets['a']._process_events({'value_throttled': 3})
    assert number.value != 3
    assert test.a == 3
    pane._widgets['a']._process_events({'value': 4})
    assert test.a == 3
    assert number.value == 4
    test.a = 5
    assert number.value == 5
    assert pane._widgets['a'].value == 5
    assert pane._widgets['a'].value_throttled == 5