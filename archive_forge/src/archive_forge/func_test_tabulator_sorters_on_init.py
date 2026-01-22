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
@pytest.mark.parametrize('col', ['index', 'int', 'float', 'str', 'date', 'datetime'])
@pytest.mark.parametrize('dir', ['ascending', 'descending'])
def test_tabulator_sorters_on_init(page, df_mixed, col, dir):
    dir_ = 'asc' if dir == 'ascending' else 'desc'
    widget = Tabulator(df_mixed, sorters=[{'field': col, 'dir': dir_}])
    serve_component(page, widget)
    sorted_header = page.locator(f'[aria-sort="{dir}"]:visible')
    expect(sorted_header).to_have_attribute('tabulator-field', col)
    ascending = dir == 'ascending'
    if col == 'index':
        expected_current_view = df_mixed.sort_index(ascending=ascending)
    else:
        expected_current_view = df_mixed.sort_values(col, ascending=ascending)
    assert widget.current_view.equals(expected_current_view)