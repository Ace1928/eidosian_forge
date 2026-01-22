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
def test_param_name_update(document, comm):

    class Test(param.Parameterized):
        a = param.Number(default=1.2, bounds=(0, 5))
    test = Test(name='A')
    test_pane = Param(test)
    model = test_pane.get_root(document, comm=comm)
    assert model.children[0].text == '<b>A</b>'
    test_pane.object = Test(name='B')
    assert model.children[0].text == '<b>B</b>'