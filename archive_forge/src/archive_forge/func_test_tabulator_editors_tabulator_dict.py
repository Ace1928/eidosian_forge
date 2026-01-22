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
def test_tabulator_editors_tabulator_dict(page, df_mixed):
    widget = Tabulator(df_mixed, editors={'str': {'type': 'textarea', 'elementAttributes': {'maxlength': '10'}}})
    serve_component(page, widget)
    cell = page.locator('text="A"')
    cell.click()
    textarea = page.locator('textarea')
    expect(textarea).to_have_count(1)
    expect(textarea).to_have_attribute('maxlength', '10')