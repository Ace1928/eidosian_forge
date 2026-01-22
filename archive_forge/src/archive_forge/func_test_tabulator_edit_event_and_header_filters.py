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
def test_tabulator_edit_event_and_header_filters(page):
    df = pd.DataFrame({'col1': list('aaabcd'), 'col2': list('ABCDEF')})
    widget = Tabulator(df, header_filters={'col1': {'type': 'input', 'func': 'like'}})
    values = []
    widget.on_edit(lambda e: values.append((e.column, e.row, e.old, e.value)))
    serve_component(page, widget)
    str_header = page.locator('input[type="search"]')
    str_header.click()
    str_header.fill('a')
    str_header.press('Enter')
    cell = page.locator('text="B"')
    cell.click()
    editable_cell = page.locator('input[type="text"]')
    editable_cell.fill('BB')
    editable_cell.press('Enter')
    wait_until(lambda: len(values) == 1, page)
    assert values[0] == ('col2', 1, 'B', 'BB')
    assert df['col2'][1] == 'BB'
    assert widget.value.equals(df)
    assert widget.current_view.equals(df.query('col1 == "a"'))