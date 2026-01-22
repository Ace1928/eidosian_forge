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
def test_switch_param_subobject(document, comm):

    class Test(param.Parameterized):
        a = param.ObjectSelector()
    o1 = Test(name='Subobject 1')
    o2 = Test(name='Subobject 2')
    Test.param['a'].objects = [o1, o2, 3]
    test = Test(a=o1, name='Nested')
    test_pane = Param(test)
    model = test_pane.get_root(document, comm=comm)
    toggle = model.children[1].children[1]
    assert isinstance(toggle, Toggle)
    test_pane._widgets['a'][1].value = True
    assert len(model.children) == 3
    _, _, subpanel = test_pane.layout.objects
    col = model.children[2]
    assert isinstance(col, BkColumn)
    assert len(col.children) == 2
    div, row = col.children
    assert div.text == '<b>Subobject 1</b>'
    assert isinstance(row.children[0], Select)
    test_pane._widgets['a'][0].value = o2
    _, _, subpanel = test_pane.layout.objects
    col = model.children[2]
    assert isinstance(col, BkColumn)
    assert len(col.children) == 2
    div, row = col.children
    assert div.text == '<b>Subobject 2</b>'
    assert isinstance(row.children[0], Select)
    test_pane._widgets['a'][1].value = False
    assert len(model.children) == 2
    assert subpanel._models == {}