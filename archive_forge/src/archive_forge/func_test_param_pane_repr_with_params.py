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
def test_param_pane_repr_with_params(document, comm):

    class Test(param.Parameterized):
        a = param.Number()
        b = param.Number()
    assert repr(Param(Test(), parameters=['a'])) == "Param(Test, parameters=['a'])"
    test_pane = Param(Test(), parameters=['a'], name='Another')
    assert repr(test_pane) == "Param(Test, name='Another', parameters=['a'])"