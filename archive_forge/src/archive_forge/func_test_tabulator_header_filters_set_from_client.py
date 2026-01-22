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
def test_tabulator_header_filters_set_from_client(page, df_mixed):
    header_filters = {'float': {'type': 'number', 'func': '>=', 'placeholder': 'Placeholder float'}, 'str': {'type': 'input', 'func': 'like', 'placeholder': 'Placeholder str'}}
    widget = Tabulator(df_mixed, header_filters=header_filters)
    serve_component(page, widget)
    number_header = page.locator('input[type="number"]')
    number_header.click()
    val, cmp, col = ('0', '>=', 'float')
    number_header.fill(val)
    number_header.press('Enter')
    query1 = f'{col} {cmp} {val}'
    expected_filter_df = df_mixed.query(query1)
    expected_filter1 = {'field': col, 'type': cmp, 'value': val}
    expect(page.locator('.tabulator-row')).to_have_count(len(expected_filter_df))
    wait_until(lambda: widget.filters == [expected_filter1], page)
    wait_until(lambda: widget.current_view.equals(expected_filter_df), page)
    str_header = page.locator('input[type="search"]')
    str_header.click()
    val, cmp, col = ('A', 'like', 'str')
    str_header.fill(val)
    str_header.press('Enter')
    query2 = f'{col} == {val!r}'
    expected_filter_df = df_mixed.query(f'{query1} and {query2}')
    expected_filter2 = {'field': col, 'type': cmp, 'value': val}
    expect(page.locator('.tabulator-row')).to_have_count(len(expected_filter_df))
    wait_until(lambda: widget.filters == [expected_filter1, expected_filter2], page)
    wait_until(lambda: widget.current_view.equals(expected_filter_df), page)