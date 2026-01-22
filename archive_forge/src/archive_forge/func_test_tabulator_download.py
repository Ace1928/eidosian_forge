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
def test_tabulator_download(page, df_mixed, df_mixed_as_string):
    widget = Tabulator(df_mixed)
    serve_component(page, widget)
    table = page.locator('.tabulator')
    expect(table).to_have_text(df_mixed_as_string, use_inner_text=True)
    with page.expect_download() as download_info:
        widget.download()
    download = download_info.value
    path = download.path()
    saved_df = pd.read_csv(path, index_col='index')
    saved_df['date'] = pd.to_datetime(saved_df['date'], unit='ms')
    saved_df['date'] = [d.to_pydatetime().date() for d in saved_df['date']]
    saved_df['datetime'] = pd.to_datetime(saved_df['datetime'], unit='ms')
    saved_df.index.name = None
    pd.testing.assert_frame_equal(df_mixed, saved_df)