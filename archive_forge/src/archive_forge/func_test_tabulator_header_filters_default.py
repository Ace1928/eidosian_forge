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
@pytest.mark.parametrize('cols', [['int', 'float', 'str', 'bool'], pytest.param(['date', 'datetime'], marks=pytest.mark.xfail(reason='See https://github.com/holoviz/panel/issues/3655'))])
def test_tabulator_header_filters_default(page, df_mixed, cols):
    df_mixed = df_mixed[cols]
    widget = Tabulator(df_mixed, header_filters=True)
    serve_component(page, widget)
    expect(page.locator('.tabulator-header-filter')).to_have_count(len(cols) + 1)
    expect(page.locator('.tabulator-row')).to_have_count(len(df_mixed))
    assert widget.filters == []
    assert widget.current_view.equals(df_mixed)