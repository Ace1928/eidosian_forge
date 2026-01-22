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
def test_number_param(document, comm):

    class Test(param.Parameterized):
        a = param.Number(default=1.2, bounds=(0, 5))
    test = Test()
    test_pane = Param(test)
    model = test_pane.get_root(document, comm=comm)
    slider = model.children[1]
    assert isinstance(slider, Slider)
    assert slider.value == 1.2
    assert slider.start == 0
    assert slider.end == 5
    assert slider.step == 0.1
    assert slider.disabled == False
    test.a = 3.3
    assert slider.value == 3.3
    a_param = test.param['a']
    a_param.bounds = (0.1, 5.5)
    assert slider.start == 0.1
    assert slider.end == 5.5
    a_param.constant = True
    assert slider.disabled == True
    test_pane._cleanup(model)
    a_param.constant = False
    a_param.bounds = (-0.1, 3.8)
    test.a = 0.5
    assert slider.value == 3.3
    assert slider.start == 0.1
    assert slider.end == 5.5
    assert slider.disabled == True