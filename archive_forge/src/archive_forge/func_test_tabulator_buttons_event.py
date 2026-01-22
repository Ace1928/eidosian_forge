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
def test_tabulator_buttons_event(page, df_mixed):
    button_col_name = 'Print'
    widget = Tabulator(df_mixed, buttons={button_col_name: 'icon'})
    state = []
    expected_state = [(button_col_name, 0, None)]

    def cb(e):
        state.append((e.column, e.row, e.value))
    widget.on_click(cb)
    serve_component(page, widget)
    icon = page.locator('text=icon').first
    icon.wait_for()
    icon.click()
    wait_until(lambda: state == expected_state, page)