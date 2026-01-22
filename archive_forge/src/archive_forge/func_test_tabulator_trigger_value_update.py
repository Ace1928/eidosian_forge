from __future__ import annotations
import datetime as dt
from contextlib import contextmanager
import numpy as np
import pandas as pd
import param
import pytest
from bokeh.models.widgets.tables import (
from playwright.sync_api import expect
from panel.depends import bind
from panel.io.state import state
from panel.layout.base import Column
from panel.models.tabulator import _TABULATOR_THEMES_MAPPING
from panel.tests.util import get_ctrl_modifier, serve_component, wait_until
from panel.widgets import Select, Tabulator
def test_tabulator_trigger_value_update(page):
    nrows = 25
    df = pd.DataFrame(np.random.rand(nrows, 2), columns=['a', 'b'])
    widget = Tabulator(df)
    serve_component(page, widget)
    expect(page.locator('.tabulator-row')).to_have_count(nrows)
    widget.param.trigger('value')
    page.wait_for_timeout(200)
    expect(page.locator('.tabulator-row')).to_have_count(nrows)