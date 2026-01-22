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
def test_tabulator_frozen_rows(page):
    arr = np.array(['a'] * 10)
    arr[1] = 'X'
    arr[-2] = 'Y'
    arr[-1] = 'T'
    df = pd.DataFrame({'col': arr})
    height, width = (200, 200)
    widget = Tabulator(df, frozen_rows=[-2, 1], height=height, width=width)
    serve_component(page, widget)
    expected_text = '\n    index\n    col\n    1\n    X\n    8\n    Y\n    0\n    a\n    2\n    a\n    3\n    a\n    4\n    a\n    5\n    a\n    6\n    a\n    7\n    a\n    9\n    T\n    '
    expect(page.locator('.tabulator')).to_have_text(expected_text, use_inner_text=True)
    X_bb = page.locator('text="X"').bounding_box()
    Y_bb = page.locator('text="Y"').bounding_box()
    page.locator('text="T"').scroll_into_view_if_needed()
    page.wait_for_timeout(200)
    assert X_bb == page.locator('text="X"').bounding_box()
    assert Y_bb == page.locator('text="Y"').bounding_box()