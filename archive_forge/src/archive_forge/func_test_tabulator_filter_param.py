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
def test_tabulator_filter_param(page, df_mixed):
    widget = Tabulator(df_mixed)

    class P(param.Parameterized):
        s = param.String()
    filt_val, filt_col = ('A', 'str')
    p = P(s=filt_val)
    widget.add_filter(p.param['s'], column=filt_col)
    serve_component(page, widget)
    df_filtered = df_mixed.loc[df_mixed[filt_col] == filt_val, :]
    wait_until(lambda: widget.current_view.equals(df_filtered), page)
    expect(page.locator('.tabulator-row')).to_have_count(len(df_filtered))
    for filt_val in ['B', 'NOT']:
        p.s = filt_val
        df_filtered = df_mixed.loc[df_mixed[filt_col] == filt_val, :]
        wait_until(lambda: widget.current_view.equals(df_filtered), page)
        expect(page.locator('.tabulator-row')).to_have_count(len(df_filtered))