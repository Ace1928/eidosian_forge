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
def test_tabulator_filter_constant_list(page, df_mixed):
    widget = Tabulator(df_mixed)
    fltr, col = (['A', 'B'], 'str')
    widget.add_filter(fltr, col)
    serve_component(page, widget)
    expect(page.locator('.tabulator-row')).to_have_count(2)
    assert page.locator('text="A"').count() == 1
    assert page.locator('text="B"').count() == 1
    assert page.locator('text="C"').count() == 0
    expected_current_view = df_mixed.loc[df_mixed[col].isin(fltr), :]
    assert widget.current_view.equals(expected_current_view)