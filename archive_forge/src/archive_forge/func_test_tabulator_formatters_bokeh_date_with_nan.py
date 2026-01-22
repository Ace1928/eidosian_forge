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
def test_tabulator_formatters_bokeh_date_with_nan(page, df_mixed):
    df_mixed.loc['idx1', 'date'] = np.nan
    df_mixed.loc['idx1', 'datetime'] = np.nan
    widget = Tabulator(df_mixed, formatters={'date': DateFormatter(format='COOKIE', nan_format='nan-date'), 'datetime': DateFormatter(format='%H:%M', nan_format='nan-datetime')})
    serve_component(page, widget)
    expect(page.locator('text="10:00"')).to_have_count(1)
    assert page.locator('text="Tue, 01 Jan 2019"').count() == 1
    assert page.locator('text="nan-date"').count() == 1
    assert page.locator('text="nan-datetime"').count() == 1