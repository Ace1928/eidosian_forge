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
def test_tabulator_selection_selectable_checkbox_multiple(page, df_mixed):
    widget = Tabulator(df_mixed, selectable='checkbox')
    serve_component(page, widget)
    checkboxes = page.locator('input[type="checkbox"]')
    checkboxes.first.wait_for()
    checkboxes.nth(1).check()
    checkboxes.last.check()
    expected_selection = [0, len(df_mixed) - 1]
    for i in range(1, checkboxes.count()):
        if i - 1 in expected_selection:
            assert checkboxes.nth(i).is_checked()
        else:
            assert not checkboxes.nth(i).is_checked()
    rows = page.locator('.tabulator-row')
    for i in range(rows.count()):
        if i in expected_selection:
            assert 'tabulator-selected' in rows.nth(i).get_attribute('class')
        else:
            assert 'tabulator-selected' not in rows.nth(i).get_attribute('class')
    wait_until(lambda: widget.selection == expected_selection, page)
    expected_selected = df_mixed.iloc[expected_selection, :]
    assert widget.selected_dataframe.equals(expected_selected)