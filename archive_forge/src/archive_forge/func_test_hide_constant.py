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
def test_hide_constant(document, comm):

    class Test(param.Parameterized):
        a = param.Number(default=1.2, bounds=(0, 5), constant=True)
    test = Test()
    test_pane = Param(test, parameters=['a'], hide_constant=True)
    model = test_pane.get_root(document, comm=comm)
    slider = model.children[1]
    assert not slider.visible
    test.param.a.constant = False
    assert slider.visible