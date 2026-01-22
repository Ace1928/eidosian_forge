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
@pytest.mark.xfail(reason='See https://github.com/holoviz/panel/issues/3664')
def test_tabulator_selection_sorters_on_init(page, df_mixed):
    widget = Tabulator(df_mixed, sorters=[{'field': 'int', 'dir': 'desc'}])
    serve_component(page, widget)
    last_index = df_mixed.index[-1]
    cell = page.locator(f'text="{last_index}"')
    cell.click()
    wait_until(lambda: widget.selection == [len(df_mixed) - 1], page)
    expected_selected = df_mixed.loc[[last_index], :]
    assert widget.selected_dataframe.equals(expected_selected)