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
def test_tabulator_header_filter_always_visible(page, df_mixed):
    col_name = 'newcol'
    df_mixed[col_name] = 'on'
    widget = Tabulator(df_mixed, header_filters={col_name: {'type': 'input', 'func': 'like'}})
    serve_component(page, widget)
    header = page.locator('input[type="search"]')
    expect(header).to_have_count(1)
    header.click()
    header.fill('off')
    header.press('Enter')
    wait_until(lambda: widget.current_view.empty, page)
    header = page.locator('input[type="search"]')
    expect(header).to_have_count(1)