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
def test_numberinput_bounds():

    class Test(param.Parameterized):
        num = param.Number(default=5, bounds=(0, 5))
    test = Test()
    p = Param(test, widgets={'num': NumberInput})
    numinput = p.layout[1]
    assert numinput.start == 0
    assert numinput.end == 5