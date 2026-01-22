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
def test_tabulator_buttons_display(page, df_mixed):
    nrows, ncols = df_mixed.shape
    icon_text = 'icon'
    widget = Tabulator(df_mixed, buttons={'Print': icon_text})
    serve_component(page, widget)
    expected_ncols = ncols + 3
    cols = page.locator('.tabulator-col')
    expect(cols).to_have_count(expected_ncols)
    button_col_idx = expected_ncols - 1
    assert not cols.nth(button_col_idx).get_attribute('tabulator-field')
    assert cols.nth(button_col_idx).inner_text() == '\xa0'
    assert cols.nth(button_col_idx).is_visible()
    icons = page.locator(f'text="{icon_text}"')
    assert icons.all_inner_texts() == [icon_text] * nrows
    for i in range(icons.count()):
        assert 'text-align: center' in icons.nth(i).get_attribute('style')