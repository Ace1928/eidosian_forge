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
def test_get_root_tabs(document, comm):

    class Test(param.Parameterized):
        pass
    test = Test()
    test_pane = Param(test, expand_layout=Tabs)
    model = test_pane.get_root(document, comm=comm)
    assert isinstance(model, BkTabs)
    assert len(model.tabs) == 1
    box = model.tabs[0].child
    assert isinstance(box, BkColumn)
    assert len(box.children) == 0