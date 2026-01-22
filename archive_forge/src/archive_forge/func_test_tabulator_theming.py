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
@pytest.mark.parametrize('theme', Tabulator.param['theme'].objects)
def test_tabulator_theming(page, df_mixed, df_mixed_as_string, theme):
    responses = []
    page.on('response', lambda response: responses.append(response))
    widget = Tabulator(df_mixed, theme=theme)
    serve_component(page, widget)
    table = page.locator('.pnx-tabulator.tabulator')
    expect(table).to_have_text(df_mixed_as_string, use_inner_text=True)
    found = False
    theme = _TABULATOR_THEMES_MAPPING.get(theme, theme)
    for response in responses:
        base = response.url.split('/')[-1]
        if base.startswith(('tabulator.min.css', f'tabulator_{theme}.min.css')):
            found = True
            break
    assert found
    assert response.status