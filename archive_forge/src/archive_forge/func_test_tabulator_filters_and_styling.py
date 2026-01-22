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
def test_tabulator_filters_and_styling(page, df_mixed):
    df_styled = df_mixed.style.apply(highlight_max, subset=['int'])
    select = Select(options=[None, 'A', 'B', 'C', 'D'], size=5)
    table = Tabulator(df_styled)
    table.add_filter(select, 'str')
    layout = Column(select, table)
    serve_component(page, layout)
    page.locator('option').nth(1).click()
    page.locator('option').nth(0).click()
    max_int = df_mixed['int'].max()
    max_cell = page.locator('.tabulator-cell', has=page.locator(f'text="{max_int}"'))
    expect(max_cell).to_have_count(1)
    expect(max_cell).to_have_css('background-color', _color_mapping['yellow'])