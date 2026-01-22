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
def test_tabulator_header_filters_multiselect(page, df_mixed):
    header_filters = {'str': {'type': 'list', 'func': 'in', 'valuesLookup': True, 'autocomplete': False, 'multiselect': True}}
    widget = Tabulator(df_mixed, header_filters=header_filters, widths=dict(str=200))
    serve_component(page, widget)
    str_header = page.locator('input[type="search"]')
    str_header.click()
    cmp, col = ('in', 'str')
    val = ['A', 'D']
    for v in val:
        item = page.locator(f'.tabulator-edit-list-item:has-text("{v}")')
        item.click()
    page.wait_for_timeout(200)
    page.locator('text="idx0"').click()
    expected_filter_df = df_mixed.query(f'{col} {cmp} {val}')
    expected_filter = {'field': col, 'type': cmp, 'value': val}
    expect(page.locator('.tabulator-row')).to_have_count(len(expected_filter_df))
    wait_until(lambda: widget.filters == [expected_filter], page)
    wait_until(lambda: widget.current_view.equals(expected_filter_df), page)