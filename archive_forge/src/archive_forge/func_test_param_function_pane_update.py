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
def test_param_function_pane_update(document, comm):
    test = View()
    objs = {0: HTML('012'), 1: HTML('123')}

    @param.depends(test.param.a)
    def view(a):
        return objs[a]
    pane = panel(view)
    inner_pane = pane._pane
    assert inner_pane is objs[0]
    assert inner_pane.object is objs[0].object
    assert not pane._internal
    test.a = 1
    assert pane._pane is not inner_pane
    assert not pane._internal
    objs[0].param.watch(print, ['object'])
    test.a = 0
    assert pane._pane is inner_pane
    assert not pane._internal