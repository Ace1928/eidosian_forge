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
def test_boolean_param(document, comm):

    class Test(param.Parameterized):
        a = param.Boolean(default=False)
    test = Test()
    test_pane = Param(test)
    model = test_pane.get_root(document, comm=comm)
    checkbox = model.children[1]
    assert isinstance(checkbox, BkCheckbox)
    assert checkbox.label == 'A'
    assert checkbox.active == False
    assert checkbox.disabled == False
    test.a = True
    assert checkbox.active == True
    a_param = test.param['a']
    a_param.constant = True
    assert checkbox.disabled == True
    test_pane._cleanup(model)
    a_param.constant = False
    test.a = False
    assert checkbox.active == True
    assert checkbox.disabled == True