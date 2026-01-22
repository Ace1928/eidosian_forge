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
def test_tabulator_frozen_columns_with_positions(page, df_mixed):
    widths = 100
    width = int((df_mixed.shape[1] + 1) * widths / 2)
    frozen_cols = {'float': 'left', 'int': 'right'}
    widget = Tabulator(df_mixed, frozen_columns=frozen_cols, width=width, widths=widths)
    serve_component(page, widget)
    expected_text = '\n    float\n    index\n    str\n    bool\n    date\n    datetime\n    int\n    3.14\n    idx0\n    A\n    true\n    2019-01-01\n    2019-01-01 10:00:00\n    1\n    6.28\n    idx1\n    B\n    true\n    2020-01-01\n    2020-01-01 12:00:00\n    2\n    9.42\n    idx2\n    C\n    true\n    2020-01-10\n    2020-01-10 13:00:00\n    3\n    -2.45\n    idx3\n    D\n    false\n    2019-01-10\n    2020-01-15 13:00:00\n    4\n    '
    table = page.locator('.pnx-tabulator.tabulator')
    expect(table).to_have_text(expected_text, use_inner_text=True)
    float_bb = page.locator('text="float"').bounding_box()
    int_bb = page.locator('text="int"').bounding_box()
    str_bb = page.locator('text="str"').bounding_box()
    assert float_bb['x'] < int_bb['x']
    assert str_bb['x'] < int_bb['x']
    page.locator('text="2019-01-01 10:00:00"').scroll_into_view_if_needed()
    wait_until(lambda: page.locator('text="str"').bounding_box()['x'] < str_bb['x'], page)
    assert float_bb == page.locator('text="float"').bounding_box()
    assert int_bb == page.locator('text="int"').bounding_box()