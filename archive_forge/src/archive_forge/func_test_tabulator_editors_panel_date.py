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
def test_tabulator_editors_panel_date(page, df_mixed):
    widget = Tabulator(df_mixed, editors={'date': 'date'})
    serve_component(page, widget)
    cell = page.locator('text="2019-01-01"')
    cell.click()
    cell_edit = page.locator('input[type="date"]')
    new_date = '1980-01-01'
    cell_edit.fill(new_date)
    page.locator('input[type="date"]').press('Enter')
    expect(page.locator(f'text="{new_date}"')).to_have_count(1)
    new_date = dt.datetime.strptime(new_date, '%Y-%m-%d').date()
    assert new_date in widget.value['date'].tolist()
    cell = page.locator(f'text="{new_date}"')
    cell.click()
    cell_edit = page.locator('input[type="date"]')
    new_date2 = '1990-01-01'
    cell_edit.fill(new_date2)
    page.locator('input[type="date"]').press('Escape')
    expect(page.locator(f'text="{new_date2}"')).to_have_count(0)
    new_date2 = dt.datetime.strptime(new_date2, '%Y-%m-%d').date()
    assert new_date2 not in widget.value['date'].tolist()