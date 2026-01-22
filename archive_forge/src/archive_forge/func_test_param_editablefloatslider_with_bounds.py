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
def test_param_editablefloatslider_with_bounds():

    class Test(param.Parameterized):
        i = param.Number(default=1, softbounds=(1, 5), bounds=(0, 10))
    t = Test()
    w = EditableFloatSlider.from_param(t.param.i)
    if Version(param.__version__) > Version('2.0.0a2'):
        msg = "Number parameter 'EditableFloatSlider.value' must be at least 0, not -1"
    else:
        msg = "Parameter 'value' must be at least 0, not -1"
    with pytest.raises(ValueError, match=msg):
        w.value = -1
    assert w.value == 1