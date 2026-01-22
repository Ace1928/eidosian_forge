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
def test_tabulator_loading_no_horizontal_rescroll(page, df_mixed):
    widths = 100
    width = int((df_mixed.shape[1] + 1) * widths / 2)
    df_mixed['Target'] = 'target'
    widget = Tabulator(df_mixed, width=width, widths=widths)
    serve_component(page, widget)
    cell = page.locator('text="target"').first
    page.wait_for_timeout(200)
    cell.scroll_into_view_if_needed()
    page.wait_for_timeout(200)
    bb = page.locator('text="Target"').bounding_box()
    widget.loading = True
    page.wait_for_timeout(200)
    widget.loading = False
    page.wait_for_timeout(400)
    assert bb == page.locator('text="Target"').bounding_box()