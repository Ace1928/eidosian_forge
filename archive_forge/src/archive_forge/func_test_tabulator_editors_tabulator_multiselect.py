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
def test_tabulator_editors_tabulator_multiselect(page, exception_handler_accumulator):
    df = pd.DataFrame({'tags': ['', '', '']}, index=['foo1', 'foo2', 'foo3'])
    tabulator_editors = {'tags': {'type': 'list', 'values': ['red', 'green', 'blue', 'orange'], 'multiselect': True}}
    widget = Tabulator(value=df, editors=tabulator_editors)
    clicks = []
    widget.on_click(clicks.append)
    serve_component(page, widget)
    cell = page.locator('.tabulator-cell:visible').nth(3)
    cell.click()
    val = ['red', 'blue']
    for v in val:
        item = page.locator(f'.tabulator-edit-list-item:has-text("{v}")')
        item.click()
    page.wait_for_timeout(200)
    page.locator('text="foo1"').click()
    cell.click()
    val = ['red', 'blue']
    for v in val:
        item = page.locator(f'.tabulator-edit-list-item:has-text("{v}")')
        item.click()
    page.wait_for_timeout(200)
    page.locator('text="foo1"').click()
    assert not exception_handler_accumulator