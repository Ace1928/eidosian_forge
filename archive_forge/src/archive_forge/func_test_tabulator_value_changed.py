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
def test_tabulator_value_changed(page, df_mixed):
    widget = Tabulator(df_mixed)
    serve_component(page, widget)
    df_mixed.loc['idx0', 'str'] = 'AA'
    widget.param.trigger('value')
    changed_cell = page.locator('text="AA"')
    expect(changed_cell).to_have_count(1)