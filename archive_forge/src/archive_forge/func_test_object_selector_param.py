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
def test_object_selector_param(document, comm):

    class Test(param.Parameterized):
        a = param.ObjectSelector(default='b', objects=[1, 'b', 'c'])
    test = Test()
    test_pane = Param(test)
    model = test_pane.get_root(document, comm=comm)
    select = model.children[1]
    assert isinstance(select, Select)
    assert select.options == [('1', '1'), ('b', 'b'), ('c', 'c')]
    assert select.value == 'b'
    assert select.disabled == False
    test.a = 1
    assert select.value == '1'
    a_param = test.param['a']
    a_param.objects = ['c', 'd', 1]
    assert select.options == [('c', 'c'), ('d', 'd'), ('1', '1')]
    a_param.constant = True
    assert select.disabled == True
    test_pane._cleanup(model)
    a_param.constant = False
    a_param.objects = [1, 'c', 'd']
    test.a = 'd'
    assert select.value == '1'
    assert select.options == [('c', 'c'), ('d', 'd'), ('1', '1')]
    assert select.disabled == True