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
def test_param_widget_type(document, comm):

    class Test(param.Parameterized):
        a = param.Number(default=1.2, bounds=(0, 5), label='A')
        b = param.Number(default=1.2, bounds=(0, 5), label='B')
    test = Test()
    test_pane = Param(test, widgets={'b': {'widget_type': EditableFloatSlider}})
    wa = test_pane._widgets['a']
    wb = test_pane._widgets['b']
    assert not isinstance(wa, EditableFloatSlider)
    assert isinstance(wb, EditableFloatSlider)
    assert wb.value == 1.2
    assert (wb.fixed_start, wb.fixed_end) == (0, 5)
    assert wb.name == 'B'