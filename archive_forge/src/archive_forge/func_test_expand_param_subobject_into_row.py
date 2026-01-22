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
def test_expand_param_subobject_into_row(document, comm):

    class Test(param.Parameterized):
        a = param.Parameter()
    test = Test(a=Test(name='Nested'))
    row = Row()
    test_pane = Param(test, expand_layout=row)
    layout = Row(test_pane, row)
    model = layout.get_root(document, comm=comm)
    toggle = model.children[0].children[1].children[1]
    assert isinstance(toggle, Toggle)
    test_pane._widgets['a'][1].value = True
    assert len(model.children) == 2
    subpanel = row.objects[0]
    row = model.children[1]
    assert isinstance(row, BkRow)
    assert len(row.children) == 1
    box = row.children[0]
    assert isinstance(box, BkColumn)
    assert len(box.children) == 2
    div, widget = box.children
    assert div.text == '<b>Nested</b>'
    assert isinstance(widget, BkTextInput)
    test_pane._widgets['a'][1].value = False
    assert len(row.children) == 0
    assert subpanel._models == {}