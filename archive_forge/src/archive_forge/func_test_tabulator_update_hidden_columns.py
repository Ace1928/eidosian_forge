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
def test_tabulator_update_hidden_columns(page):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3]})
    widget = Tabulator(df, hidden_columns=['a', 'b'], sizing_mode='stretch_width')
    serve_component(page, widget)
    col_a_cells = page.locator('text="3"')
    expect(col_a_cells.nth(0)).not_to_be_visible()
    expect(col_a_cells.nth(1)).not_to_be_visible()
    widget.hidden_columns = ['b']
    col_a_cells = page.locator('text="3"')
    expect(col_a_cells.nth(0)).to_be_visible()
    expect(col_a_cells.nth(1)).not_to_be_visible()
    title = page.locator('text="a"')
    cell = col_a_cells.first
    wait_until(lambda: title.bounding_box()['x'] == cell.bounding_box()['x'] and title.bounding_box()['width'] == cell.bounding_box()['width'], page)