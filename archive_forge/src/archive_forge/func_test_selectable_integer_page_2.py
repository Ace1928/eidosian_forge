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
def test_selectable_integer_page_2(self, page):
    self.goto_page(page, 2)
    rows = self.get_rows(page)
    with self.hold_down_ctrl(page):
        rows.nth(0).click()
        rows.nth(1).click()
    self.check_selected(page, [10, 11])
    with self.hold_down_ctrl(page):
        rows.nth(2).click()
    self.check_selected(page, [11, 12])
    self.goto_page(page, 1)
    self.check_selected(page, [11, 12], 0)