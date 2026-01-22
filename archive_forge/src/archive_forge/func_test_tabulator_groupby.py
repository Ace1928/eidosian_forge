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
def test_tabulator_groupby(page):
    df = pd.DataFrame({'cat1': ['A', 'B', 'A', 'A', 'B', 'B', 'B'], 'cat2': ['X', 'X', 'X', 'X', 'Y', 'Y', 'Y'], 'value': list(range(7))})
    widget = Tabulator(df, groupby=['cat1', 'cat2'])
    serve_component(page, widget)
    expected_text = '\n    index\n    cat1\n    cat2\n    value\n    cat1: A, cat2: X(3 items)\n    0\n    A\n    X\n    0\n    2\n    A\n    X\n    2\n    3\n    A\n    X\n    3\n    cat1: B, cat2: X(1 item)\n    1\n    B\n    X\n    1\n    cat1: B, cat2: Y(3 items)\n    4\n    B\n    Y\n    4\n    5\n    B\n    Y\n    5\n    6\n    B\n    Y\n    6\n    '
    expect(page.locator('.tabulator')).to_have_text(expected_text, use_inner_text=True)
    expect(page.locator('.tabulator-group')).to_have_count(3)