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
def test_tabulator_formatters_bokeh_html_multiple_columns(page, df_mixed):
    htmlfmt = HTMLTemplateFormatter(template='<p class="html-format"><%= str %> <%= bool %></p>')
    widget = Tabulator(df_mixed, formatters={'str': htmlfmt})
    serve_component(page, widget)
    cells = page.locator('.tabulator-cell .html-format')
    expect(cells).to_have_count(len(df_mixed))
    for i, (_, row) in enumerate(df_mixed.iterrows()):
        expect(cells.nth(i)).to_have_text(f'{row['str']} {str(row['bool']).lower()}')