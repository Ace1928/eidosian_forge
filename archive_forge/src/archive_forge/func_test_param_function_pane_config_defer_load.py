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
def test_param_function_pane_config_defer_load():
    app = ParameterizedMock()
    test = ParamMethod(app.click_view)
    assert test.defer_load == config.defer_load
    config.defer_load = not config.defer_load
    app = ParameterizedMock()
    test = ParamMethod(app.click_view)
    assert test.defer_load == config.defer_load
    config.defer_load = not config.defer_load
    app = ParameterizedMock()
    test = ParamMethod(app.click_view)
    assert test.defer_load == config.defer_load
    config.defer_load = config.param.defer_load.default