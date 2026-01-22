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
def test_param_event_parameter(document, comm):
    l = []

    class Test(param.Parameterized):
        e = param.Event()

        @param.depends('e', watch=True)
        def incr(self):
            l.append(1)
    test = Test()
    test_pane = Param(test)
    test_pane._widgets['e']._process_events({'clicks': 1})
    assert l == [1]