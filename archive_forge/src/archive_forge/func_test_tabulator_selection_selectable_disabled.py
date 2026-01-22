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
def test_tabulator_selection_selectable_disabled(page, df_mixed):
    widget = Tabulator(df_mixed, selectable=False)
    serve_component(page, widget)
    rows = page.locator('.tabulator-row')
    c0 = page.locator('text="idx0"')
    c0.wait_for()
    c0.click()
    page.wait_for_timeout(200)
    assert widget.selection == []
    assert widget.selected_dataframe.empty
    for i in range(rows.count()):
        assert 'tabulator-selected' not in rows.nth(i).get_attribute('class')