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
@pytest.mark.parametrize('pagination', ['remote', 'local'])
def test_tabulator_edit_event_and_header_filters_same_column_pagination(page, pagination):
    df = pd.DataFrame({'values': ['A', 'A', 'B', 'B', 'B', 'B']}, index=['idx0', 'idx1', 'idx2', 'idx3', 'idx4', 'idx5'])
    widget = Tabulator(df, header_filters={'values': {'type': 'input', 'func': 'like'}}, pagination=pagination, page_size=2)
    values = []
    widget.on_edit(lambda e: values.append((e.column, e.row, e.old, e.value)))
    serve_component(page, widget)
    header = page.locator('input[type="search"]')
    header.click()
    header.fill('B')
    header.press('Enter')
    wait_until(lambda: widget.current_view.equals(df[df['values'] == 'B']))
    cell = page.locator('text="B"').first
    cell.click()
    editable_cell = page.locator('input[type="text"]')
    editable_cell.fill('Q')
    editable_cell.press('Enter')
    wait_until(lambda: len(values) == 1, page)
    assert values[-1] == ('values', 2, 'B', 'Q')
    assert df.at['idx2', 'values'] == 'Q'
    assert len(widget.current_view) == 4
    page.locator('text="Last"').click()
    page.wait_for_timeout(200)
    expect(page.locator('.tabulator-row')).to_have_count(2)
    cell = page.locator('text="B"').nth(1)
    cell.click()
    editable_cell = page.locator('input[type="text"]')
    editable_cell.fill('X')
    editable_cell.press('Enter')
    wait_until(lambda: len(values) == 2, page)
    assert values[-1] == ('values', len(df) - 1, 'B', 'X')
    assert df.at['idx5', 'values'] == 'X'
    assert len(widget.current_view) == 4
    cell = page.locator('text="X"')
    cell.click()
    editable_cell = page.locator('input[type="text"]')
    editable_cell.fill('Y')
    editable_cell.press('Enter')
    wait_until(lambda: len(values) == 3, page)
    assert values[-1] == ('values', len(df) - 1, 'X', 'Y')
    assert df.at['idx5', 'values'] == 'Y'
    assert len(widget.current_view) == 4
    cell = page.locator('text="B"')
    cell.click()
    editable_cell = page.locator('input[type="text"]')
    editable_cell.fill('Z')
    editable_cell.press('Enter')
    wait_until(lambda: len(values) == 4, page)
    assert values[-1] == ('values', len(df) - 2, 'B', 'Z')
    assert df.at['idx4', 'values'] == 'Z'
    assert len(widget.current_view) == 4