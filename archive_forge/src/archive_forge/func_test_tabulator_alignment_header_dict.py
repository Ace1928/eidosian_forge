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
def test_tabulator_alignment_header_dict(page, df_mixed):
    halign = {'int': 'left'}
    widget = Tabulator(df_mixed, header_align=halign)
    serve_component(page, widget)
    for col, align in halign.items():
        expect(page.locator(f'text="{col}"')).to_have_css('text-align', align)