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
@pytest.mark.parametrize('pagination', ['remote', 'local'])
def test_tabulator_sorters_pagination(page, df_mixed, pagination):
    widget = Tabulator(df_mixed, pagination=pagination, page_size=2)
    serve_component(page, widget)
    s = page.locator('.tabulator-col', has_text='str').locator('.tabulator-col-sorter')
    s.click()
    page.wait_for_timeout(100)
    s.click()
    sheader = page.locator('[aria-sort="descending"]:visible')
    expect(sheader).to_have_count(1)
    assert sheader.get_attribute('tabulator-field') == 'str'
    expected_sorted_df = df_mixed.sort_values('str', ascending=False)
    wait_until(lambda: widget.current_view.equals(expected_sorted_df), page)
    page.locator('text="Next"').click()
    page.wait_for_timeout(200)
    wait_until(lambda: widget.current_view.equals(expected_sorted_df), page)