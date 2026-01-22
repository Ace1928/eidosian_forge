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
def test_tabulator_filter_bound_function(page, df_mixed):
    widget = Tabulator(df_mixed)

    def filt_(df, val):
        return df[df['str'] == val]
    filt_val = 'A'
    w_filter = Select(value='A', options=['A', 'B', ''])
    widget.add_filter(bind(filt_, val=w_filter))
    serve_component(page, widget)
    df_filtered = filt_(df_mixed, w_filter.value)
    wait_until(lambda: widget.current_view.equals(df_filtered), page)
    expect(page.locator('.tabulator-row')).to_have_count(len(df_filtered))
    for filt_val in w_filter.options[1:]:
        w_filter.value = filt_val
        df_filtered = filt_(df_mixed, filt_val)
        wait_until(lambda: widget.current_view.equals(df_filtered), page)
        expect(page.locator('.tabulator-row')).to_have_count(len(df_filtered))