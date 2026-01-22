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
def test_sorted_func():

    class MyClass(param.Parameterized):
        valueb = param.Integer(label='bac')
        valuez = param.String(label='acb')
        valuea = param.Integer(label='cba')
    my_class = MyClass()

    def sort_func(x):
        return x[1].label[::-1]
    _, input1, input2, input3 = Param(my_class, sort=sort_func)
    assert input1.name == 'cba'
    assert input2.name == 'acb'
    assert input3.name == 'bac'