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
def test_tabulator_selection_selectable_by_default(page, df_mixed):
    widget = Tabulator(df_mixed)
    serve_component(page, widget)
    assert widget.selectable
    rows = page.locator('.tabulator-row')
    c0 = page.locator('text="idx0"')
    c0.wait_for()
    c0.click()
    wait_until(lambda: widget.selection == [0], page)
    assert 'tabulator-selected' in rows.first.get_attribute('class')
    for i in range(1, rows.count()):
        assert 'tabulator-selected' not in rows.nth(i).get_attribute('class')
    expected_selected = df_mixed.loc[['idx0'], :]
    assert widget.selected_dataframe.equals(expected_selected)