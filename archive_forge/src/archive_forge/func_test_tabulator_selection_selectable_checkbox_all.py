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
def test_tabulator_selection_selectable_checkbox_all(page, df_mixed):
    widget = Tabulator(df_mixed, selectable='checkbox')
    serve_component(page, widget)
    checkboxes = page.locator('input[type="checkbox"]')
    checkboxes.first.wait_for()
    checkboxes.first.check()
    for i in range(checkboxes.count()):
        assert checkboxes.nth(i).is_checked()
    rows = page.locator('.tabulator-row')
    for i in range(rows.count()):
        assert 'tabulator-selected' in rows.nth(i).get_attribute('class')
    wait_until(lambda: widget.selection == list(range(len(df_mixed))), page)
    assert widget.selected_dataframe.equals(df_mixed)