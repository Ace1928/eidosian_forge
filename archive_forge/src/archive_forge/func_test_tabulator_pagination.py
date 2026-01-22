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
def test_tabulator_pagination(page, df_mixed, pagination):
    page_size = 2
    widget = Tabulator(df_mixed, pagination=pagination, page_size=page_size)
    serve_component(page, widget)
    counts = count_per_page(len(df_mixed), page_size)
    i = 0
    while True:
        wait_until(lambda: widget.page == i + 1, page)
        rows = page.locator('.tabulator-row')
        expect(rows).to_have_count(counts[i])
        assert page.locator(f'[aria-label="Show Page {i + 1}"]').count() == 1
        df_page = df_mixed.iloc[i * page_size:(i + 1) * page_size]
        for idx in df_page.index:
            assert page.locator(f'text="{idx}"').count() == 1
        if i < len(counts) - 1:
            page.locator(f'[aria-label="Show Page {i + 2}"]').click()
            i += 1
        else:
            break