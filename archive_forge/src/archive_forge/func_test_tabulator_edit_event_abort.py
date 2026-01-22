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
def test_tabulator_edit_event_abort(page, df_mixed):
    widget = Tabulator(df_mixed)
    values = []
    widget.on_edit(lambda e: values.append((e.column, e.row, e.old, e.value)))
    serve_component(page, widget)
    cell = page.locator('text="3.14"')
    cell.click()
    editable_cell = page.locator('input[type="number"]')
    editable_cell.fill('0')
    editable_cell.press('Escape')
    expect(cell).to_contain_text('3.14')
    page.wait_for_timeout(200)
    assert values == []