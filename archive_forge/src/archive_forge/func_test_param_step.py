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
def test_param_step(document, comm):

    class Test(param.Parameterized):
        a = param.Number(default=1.2, bounds=(0, 5), step=0.1)
    test = Test()
    test_pane = Param(test)
    assert test_pane._widgets['a'].step == 0.1
    a_param = test.param['a']
    a_param.step = 0.25
    assert test_pane._widgets['a'].step == 0.25